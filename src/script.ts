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

  const CELL_SIZE = 0.75;
  const GRID_SIZE = 64;
  const WORKGROUP_SIZE = 8;
  const STEP_RATE = 8;

  // Frame, Time, Last Render Time
  const renderInfoData = new Float32Array([0, 0, 0]);
  const renderInfoBuffer = device.createBuffer({
    label: "Render Info",
    size: renderInfoData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(renderInfoBuffer, 0, renderInfoData);

  const mouseData = new Float32Array([0, 0, 0]);
  const mouseBuffer = device.createBuffer({
    label: "Mouse",
    size: mouseData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(mouseBuffer, 0, mouseData);

  const gridSizeData = new Float32Array([GRID_SIZE, GRID_SIZE]);
  const gridSizeBuffer = device.createBuffer({
    label: "Grid Size",
    size: gridSizeData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridSizeBuffer, 0, gridSizeData);

  // Holds the current and next state of the grid as a single buffer.
  const gridStateData = new Uint32Array(GRID_SIZE * GRID_SIZE * 2).fill(0);
  const gridStateBuffer = device.createBuffer({
    label: "Grid State",
    size: gridStateData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  for (let i = 0, l = gridStateData.length; i < l; i += 1) {
    gridStateData[i] = Math.random() <= 0.1 ? 1 : 0;
  }
  device.queue.writeBuffer(gridStateBuffer, 0, gridStateData);

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

  const renderBindGroupLayoutDescriptor = {
    label: "Render Bind Group Layout",
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
        visibility:
          GPUShaderStage.VERTEX |
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
      {
        binding: 2,
        visibility:
          GPUShaderStage.VERTEX |
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
      {
        binding: 3,
        visibility:
          GPUShaderStage.VERTEX |
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  } satisfies GPUBindGroupLayoutDescriptor;

  const renderBindGroupLayout = device.createBindGroupLayout(
    renderBindGroupLayoutDescriptor
  );

  const computeBindGroupLayoutDescriptor = {
    label: "Compute Bind Group Layout",
    entries: [
      renderBindGroupLayoutDescriptor.entries[0],
      renderBindGroupLayoutDescriptor.entries[1],
      renderBindGroupLayoutDescriptor.entries[2],
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  } satisfies GPUBindGroupLayoutDescriptor;

  const computeBindGroupLayout = device.createBindGroupLayout(
    computeBindGroupLayoutDescriptor
  );

  const renderBindGroupDescriptor = {
    label: "Render Bind Group",
    layout: renderBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: renderInfoBuffer },
      },
      {
        binding: 1,
        resource: { buffer: mouseBuffer },
      },
      {
        binding: 2,
        resource: { buffer: gridSizeBuffer },
      },
      {
        binding: 3,
        resource: { buffer: gridStateBuffer },
      },
    ],
  } satisfies GPUBindGroupDescriptor;

  const renderBindGroup = device.createBindGroup(renderBindGroupDescriptor);

  const computeBindGroupDescriptor = {
    label: "Compute Bind Group",
    layout: computeBindGroupLayout,
    entries: renderBindGroupDescriptor.entries,
  } satisfies GPUBindGroupDescriptor;

  const computeBindGroup = device.createBindGroup(computeBindGroupDescriptor);

  const renderPipelineLayout = device.createPipelineLayout({
    label: "Render Pipeline Layout",
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const computePipelineLayout = device.createPipelineLayout({
    label: "Compute Pipeline Layout",
    bindGroupLayouts: [computeBindGroupLayout],
  });

  const renderShaderModule = device.createShaderModule({
    label: "Render Shader Module",
    code: /*wgsl*/ `
      struct RenderInfo {
        frame: f32,
        time: f32,
        lastRenderTime: f32,
      };

      @group(0) @binding(0) var<uniform> renderInfo: RenderInfo;
      @group(0) @binding(1) var<uniform> mouse: vec3f;
      @group(0) @binding(2) var<uniform> gridSize: vec2f;
      @group(0) @binding(3) var<storage> gridState: array<u32>;

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
        let position = (input.position * f32(gridState[input.index]) + 1) / gridSize - 1 + offset;

        var output: VertexOutput;
        output.position = vec4f(position, 0, 1);
        output.cell = cell;
        return output;
      }

      const keyframeLength = 1000.0;

      const keyframes = array(
        vec4f(1, 0, 0, 1),
        vec4f(1, 1, 0, 1),
        vec4f(0, 1, 0, 1),
        vec4f(0, 1, 1, 1),
        vec4f(0, 0, 1, 1),
        vec4f(0, 1, 1, 1),
        vec4f(0, 1, 0, 1),
        vec4f(1, 1, 0, 1),
        vec4f(1, 0, 0, 1),
      );

      const keyframesTotal = 9.0;

      @fragment
      fn fmain(input: VertexOutput) -> @location(0) vec4f {
        let t = renderInfo.time;

        let keyframeIndex = u32(floor(t / keyframeLength) % keyframesTotal);
        let keyframeProgress = fract(t / keyframeLength);

        return mix(keyframes[keyframeIndex], keyframes[keyframeIndex + 1], keyframeProgress);
      }
    `,
  });

  const computeShaderModule = device.createShaderModule({
    label: "Compute Shader Module",
    code: /*wgsl*/ `
      struct RenderInfo {
        frame: f32,
        time: f32,
        lastRenderTime: f32,
      };

      @group(0) @binding(0) var<uniform> renderInfo: RenderInfo;
      @group(0) @binding(1) var<uniform> mouse: vec3f;
      @group(0) @binding(2) var<uniform> gridSize: vec2f;
      @group(0) @binding(3) var<storage, read_write> gridState: array<u32>;

      fn getIndex(x: u32, y: u32) -> u32 {
        return (y % u32(gridSize.y)) * u32(gridSize.x) + (x % u32(gridSize.x));
      }

      fn getState(x: u32, y: u32) -> u32 {
        return gridState[getIndex(x, y)];
      }

      fn setNextStateAt(index: u32, state: u32) {
        let offset = u32(gridSize.x * gridSize.y);
        gridState[offset + index] = state;
      }

      @compute
      @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
      fn cmain(@builtin(global_invocation_id) cell: vec3u) {
        let i = getIndex(cell.x, cell.y);

        if ((u32(mouse.z) & 1) == 1) {
          let mouseGridPosition = vec2u(
            u32(floor((mouse.x + 1) / 2 * gridSize.x)),
            u32(floor((mouse.y + 1) / 2 * gridSize.y))
          );

          if (all(mouseGridPosition == cell.xy)) {
            gridState[i] = 1;
          }

          return;
        }

        if (floor(renderInfo.frame % ${STEP_RATE}) > 0) {
          return;
        }

        let activeNeighbors = getState(cell.x + 1, cell.y + 1) + getState(cell.x + 1, cell.y) +
          getState(cell.x + 1, cell.y - 1) + getState(cell.x, cell.y - 1) +
          getState(cell.x - 1, cell.y - 1) + getState(cell.x - 1, cell.y) +
          getState(cell.x - 1, cell.y + 1) + getState(cell.x, cell.y + 1);

        switch activeNeighbors {
          case 2: {
            setNextStateAt(i, gridState[i]);
          }
          case 3: {
            setNextStateAt(i, 1);
          }
          default: {
            setNextStateAt(i, 0);
          }
        }

        let offset = u32(gridSize.x * gridSize.y);
        for (var i = 0u; i < offset; i += 1) {
          gridState[i] = gridState[offset + i];
        }
      }
    `,
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Render Pipeline",
    layout: renderPipelineLayout,
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

  const computePipeline = device.createComputePipeline({
    label: "Simulation Pipeline",
    layout: computePipelineLayout,
    compute: {
      module: computeShaderModule,
    },
  });

  const mouse = { x: 0, y: 0, pressed: 0 };

  canvas.addEventListener("mousedown", (event) => {
    event.preventDefault();

    mouse.pressed = event.buttons;
  });

  canvas.addEventListener("mouseup", (event) => {
    event.preventDefault();

    mouse.pressed = event.buttons;
  });

  canvas.addEventListener("mousemove", (event) => {
    mouse.x = (event.offsetX / canvas.width) * 2 - 1;
    mouse.y = (event.offsetY / canvas.height) * -2 + 1;
  });

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  let lastRenderTime = 0;
  let frame = 0;

  function render(time: number) {
    const renderTimeDelta = time - lastRenderTime;
    lastRenderTime = time;

    const encoder = device.createCommandEncoder();

    renderInfoData.set([frame, time, lastRenderTime]);
    device.queue.writeBuffer(renderInfoBuffer, 0, renderInfoData);

    mouseData.set([mouse.x, mouse.y, mouse.pressed]);
    device.queue.writeBuffer(mouseBuffer, 0, mouseData);

    const computePass = encoder.beginComputePass();
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    computePass.end();

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
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(cellVertexData.length / 2, GRID_SIZE * GRID_SIZE);
    renderPass.end();

    device.queue.submit([encoder.finish()]);

    frame += 1;
    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

main().catch((err: Error) => {
  alert(err.message);
});
