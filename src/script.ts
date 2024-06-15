async function getWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported.");
  }

  const canvas = document.createElement("canvas");
  document.body.appendChild(canvas);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No GPUAdapter found.");
  }

  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("WebGPU context failed.");
  }

  context.configure({
    device,
    format,
  });

  return { canvas, context, device, format } as const;
}

async function main() {
  const { canvas, context, device, format } = await getWebGPU();

  canvas.height = 640;
  canvas.width = canvas.height;

  const CELL_SIZE = 0.5;
  const GRID_SIZE = 64;
  const WORKGROUP_SIZE = 8;
  const STEP_RATE = 500;

  const gridSizeData = new Float32Array([GRID_SIZE, GRID_SIZE]);
  const gridSizeBuffer = device.createBuffer({
    label: "Grid Size",
    size: gridSizeData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridSizeBuffer, 0, gridSizeData);

  const gridStateData = new Uint32Array(GRID_SIZE * GRID_SIZE).fill(0);
  const gridStateBuffers = [
    device.createBuffer({
      label: "Grid State",
      size: gridStateData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      label: "Grid State",
      size: gridStateData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
  ];
  device.queue.writeBuffer(gridStateBuffers[1], 0, gridStateData);
  for (let i = 0; i < gridStateData.length; i += 1) {
    gridStateData[i] = Math.random() >= 0.5 ? 1 : 0;
  }
  device.queue.writeBuffer(gridStateBuffers[0], 0, gridStateData);

  const cellVertexData = new Float32Array([
    -CELL_SIZE,
    -CELL_SIZE,
    CELL_SIZE,
    -CELL_SIZE,
    CELL_SIZE,
    CELL_SIZE,
    -CELL_SIZE,
    -CELL_SIZE,
    CELL_SIZE,
    CELL_SIZE,
    -CELL_SIZE,
    CELL_SIZE,
  ]);
  const cellVertexBuffer = device.createBuffer({
    label: "Cell Vertex",
    size: cellVertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cellVertexBuffer, 0, cellVertexData);

  const bindGroupLayout = device.createBindGroupLayout({
    label: "Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility:
          GPUShaderStage.VERTEX |
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  const bindGroups = [
    device.createBindGroup({
      label: "Bind Group A",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: gridSizeBuffer },
        },
        {
          binding: 1,
          resource: { buffer: gridStateBuffers[0] },
        },
        {
          binding: 2,
          resource: { buffer: gridStateBuffers[1] },
        },
      ],
    }),

    device.createBindGroup({
      label: "Bind Group B",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: gridSizeBuffer },
        },
        {
          binding: 1,
          resource: { buffer: gridStateBuffers[1] },
        },
        {
          binding: 2,
          resource: { buffer: gridStateBuffers[0] },
        },
      ],
    }),
  ];

  const pipelinesSharedLayout = device.createPipelineLayout({
    label: "Pipelines Shared Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const renderShaderModule = device.createShaderModule({
    label: "Render Shader Module",
    code: /*wgsl*/ `
      @group(0) @binding(0) var<uniform> gridSize: vec2f;
      @group(0) @binding(1) var<storage> gridCurrentState: array<u32>;

      struct VertexInput {
        @location(0) position: vec2f,
        @builtin(instance_index) index: u32,
      };

      struct VertexOutput {
        @location(0) cell: vec2f,
        @builtin(position) position: vec4f,
      };

      @vertex
      fn vmain(input: VertexInput) -> VertexOutput {
        let i = f32(input.index);
        let cell = vec2f(i % gridSize.x, floor(i / gridSize.x));
        let offset = cell / gridSize * 2;
        let position = (input.position * f32(gridCurrentState[input.index]) + 1) / gridSize - 1 + offset;

        var output: VertexOutput;
        output.position = vec4f(position, 0, 1);
        output.cell = cell;
        return output;
      }

      @fragment
      fn fmain(input: VertexOutput) -> @location(0) vec4f {
        let index = input.cell / gridSize;
        return vec4f(index.xy, 1 - index.x, 1) + vec4f(0.25);
      }
    `,
  });

  const simulationShaderModule = device.createShaderModule({
    label: "Simulation Shader Module",
    code: /*wgsl*/ `
      @group(0) @binding(0) var<uniform> gridSize: vec2f;
      @group(0) @binding(1) var<storage> gridCurrentState: array<u32>;
      @group(0) @binding(2) var<storage, read_write> gridNextState: array<u32>;

      fn indexOf(x: u32, y: u32) -> u32 {
        return (y % u32(gridSize.y)) * u32(gridSize.x) + (x % u32(gridSize.x));
      }

      fn stateOf(x: u32, y: u32) -> u32 {
        return gridCurrentState[indexOf(x, y)];
      }

      @compute
      @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
      fn cmain(@builtin(global_invocation_id) cell: vec3u) {
        let n = stateOf(cell.x + 1, cell.y + 1) + stateOf(cell.x + 1, cell.y) +
          stateOf(cell.x + 1, cell.y - 1) + stateOf(cell.x, cell.y - 1) +
          stateOf(cell.x - 1, cell.y - 1) + stateOf(cell.x - 1, cell.y) +
          stateOf(cell.x - 1, cell.y + 1) + stateOf(cell.x, cell.y + 1);

        let i = indexOf(cell.x, cell.y);

        switch n {
          case 2: {
            gridNextState[i] = gridCurrentState[i];
          }
          case 3: {
            gridNextState[i] = 1;
          }
          default: {
            gridNextState[i] = 0;
          }
        }
      }
    `,
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Render Pipeline",
    layout: pipelinesSharedLayout,
    vertex: {
      module: renderShaderModule,
      buffers: [
        {
          arrayStride: 8,
          attributes: [
            {
              format: "float32x2",
              offset: 0,
              shaderLocation: 0,
            },
          ],
        },
      ],
    },
    fragment: {
      module: renderShaderModule,
      targets: [
        {
          format: format,
        },
      ],
    },
  });

  const simulationPipeline = device.createComputePipeline({
    label: "Simulation Pipeline",
    layout: pipelinesSharedLayout,
    compute: {
      module: simulationShaderModule,
    },
  });

  let lastSimulationTime = 0;
  let lastRenderTime = 0;
  let step = 1;

  function render(time: number) {
    const simulationTimeDelta = time - lastSimulationTime;
    const renderTimeDelta = time - lastRenderTime;
    lastRenderTime = time;

    requestAnimationFrame(render);

    const encoder = device.createCommandEncoder();

    if (simulationTimeDelta >= STEP_RATE) {
      lastSimulationTime = time;
      step += 1;

      const computePass = encoder.beginComputePass();
      const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
      computePass.setPipeline(simulationPipeline);
      computePass.setBindGroup(0, bindGroups[step % 2]);
      computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
      computePass.end();
    }

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear" as const,
          storeOp: "store" as const,
        },
      ],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setVertexBuffer(0, cellVertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(cellVertexData.length / 2, GRID_SIZE * GRID_SIZE);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
  }

  requestAnimationFrame(render);
}

main();
