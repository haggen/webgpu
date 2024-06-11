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

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("WebGPU context failed.");
  }

  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
  });

  return { canvas, context, device, format };
}

async function main() {
  const { canvas, context, device, format } = await getWebGPU();

  canvas.height = 640;
  canvas.width = canvas.height;

  const CELL_SIZE = 0.5;
  const GRID_SIZE = 128;
  const WORKGROUP_SIZE = 8;
  const STEP_RATE = 50;

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

  const cellShaderModule = device.createShaderModule({
    label: "Cell Shader",
    code: /*wgsl*/ `
      @group(0) @binding(0) var<uniform> grid: vec2f;
      @group(0) @binding(1) var<storage> state: array<u32>;

      struct VertexInput {
          @location(0) position: vec2f,
          @builtin(instance_index) instance: u32,
      };

      struct VertexOutput {
          @location(0) cell: vec2f,
          @builtin(position) position: vec4f,
      };

      @vertex
      fn vmain(input: VertexInput) -> VertexOutput {
          let i = f32(input.instance);
          let cell = vec2f(i % grid.x, floor(i / grid.x));
          let offset = cell / grid * 2;
          let position = (input.position * f32(state[input.instance]) + 1) / grid - 1 + offset;

          var output: VertexOutput;
          output.position = vec4f(position, 0, 1);
          output.cell = cell;
          return output;
      }

      @fragment
      fn fmain(input: VertexOutput) -> @location(0) vec4f {
          let color = input.cell / grid;
          return vec4f(color.xy+0.5, 1.5 - color.x, 1);
      }
    `,
  });

  const simulationShaderModule = device.createShaderModule({
    label: "Simulation",
    code: /*wgsl*/ `
      @group(0) @binding(0) var<uniform> grid: vec2f;
      @group(0) @binding(1) var<storage> state_in: array<u32>;
      @group(0) @binding(2) var<storage, read_write> state_out: array<u32>;

      fn cell_index(cell: vec2u) -> u32 {
        return (cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x));
      }

      fn cell_state(x: u32, y: u32) -> u32 {
        return state_in[cell_index(vec2u(x, y))];
      }

      @compute
      @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
      fn cmain(@builtin(global_invocation_id) cell: vec3u) {
        let active_neighbors = cell_state(cell.x+1, cell.y+1) + cell_state(cell.x+1, cell.y) +
          cell_state(cell.x+1, cell.y-1) + cell_state(cell.x, cell.y-1) +
          cell_state(cell.x-1, cell.y-1) + cell_state(cell.x-1, cell.y) +
          cell_state(cell.x-1, cell.y+1) + cell_state(cell.x, cell.y+1);

        let i = cell_index(cell.xy);

        switch active_neighbors {
          case 2: {
            state_out[i] = state_in[i];
          }
          case 3: {
            state_out[i] = 1;
          }
          default: {
            state_out[i] = 0;
          }
        }
      }
    `,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility:
          GPUShaderStage.VERTEX |
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.COMPUTE,
        buffer: {},
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

  const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const cellRenderPipeline = device.createRenderPipeline({
    label: "Cell Render Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: cellShaderModule,
      entryPoint: "vmain",
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
      module: cellShaderModule,
      entryPoint: "fmain",
      targets: [
        {
          format: format,
        },
      ],
    },
  });

  const simulationPipeline = device.createComputePipeline({
    label: "Simulation Pipeline",
    layout: pipelineLayout,
    compute: {
      module: simulationShaderModule,
      entryPoint: "cmain",
    },
  });

  let step = 0;
  let lastUpdateTime = 0;

  function draw(time: number) {
    const updateTimeDelta = time - lastUpdateTime;

    const encoder = device.createCommandEncoder();

    if (updateTimeDelta >= STEP_RATE) {
      const computePass = encoder.beginComputePass();
      const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
      computePass.setPipeline(simulationPipeline);
      computePass.setBindGroup(0, bindGroups[step % 2]);
      computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
      computePass.end();

      step += 1;
      lastUpdateTime = time;
    }

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(cellRenderPipeline);
    renderPass.setVertexBuffer(0, cellVertexBuffer);
    renderPass.setBindGroup(0, bindGroups[step % 2]);
    renderPass.draw(cellVertexData.length / 2, GRID_SIZE * GRID_SIZE);
    renderPass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
}

main();
