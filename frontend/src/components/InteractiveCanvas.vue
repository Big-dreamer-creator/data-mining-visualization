<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-vue-next";

const props = defineProps({
  result: {
    type: Object,
    required: true,
  },
});

const canvasRef = ref(null);
const stageRef = ref(null);
const currentStep = ref(0);

const steps = computed(() => props.result.steps || []);
const frame = computed(() => steps.value[currentStep.value] || steps.value[0] || emptyFrame());
const labels = computed(() => props.result.labels || []);
const bounds = computed(() => props.result.coordinateSystem?.bounds || { xMin: -1, xMax: 1, yMin: -1, yMax: 1 });
const coordinateSystem = computed(() => props.result.coordinateSystem || {});

watch(
  () => props.result,
  () => reset(),
  { deep: true },
);

watch(currentStep, () => nextTick(draw));

onMounted(() => {
  window.addEventListener("resize", resize);
  resize();
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", resize);
});

function emptyFrame() {
  return { index: 0, description: "", scatter: [], backgroundGrid: null, helpers: [], annotations: [] };
}

function nextStep() {
  currentStep.value = Math.min(currentStep.value + 1, Math.max(steps.value.length - 1, 0));
}

function previousStep() {
  currentStep.value = Math.max(currentStep.value - 1, 0);
}

function reset() {
  currentStep.value = 0;
  nextTick(() => {
    resize();
    draw();
  });
}

function resize() {
  const canvas = canvasRef.value;
  const stage = stageRef.value;
  if (!canvas || !stage) return;
  const rect = stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const height = Math.max(500, Math.min(720, window.innerHeight - 220));
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(height * ratio);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${height}px`;
  canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function draw() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  paintSurface(ctx, width, height);
  paintGrid(ctx, width, height);
  paintBackground(ctx, frame.value.backgroundGrid);
  paintHelpers(ctx, frame.value.helpers || []);
  paintScatter(ctx, frame.value.scatter || []);
  paintLabels(ctx, width, height);
  paintStepBar(ctx, width, height);
}

function paintSurface(ctx, width, height) {
  ctx.fillStyle = "#f8fbff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#eef4fb";
  ctx.fillRect(0, 0, width, 44);
  ctx.fillStyle = "#172033";
  ctx.font = "13px sans-serif";
  ctx.fillText(`${props.result.algorithmName} · 后端状态帧播放`, 18, 27);
}

function paintGrid(ctx, width, height) {
  ctx.save();
  ctx.strokeStyle = "#dbe5f1";
  ctx.lineWidth = 1;
  for (let x = 52; x < width; x += 52) line(ctx, x, 44, x, height - 32);
  for (let y = 82; y < height - 32; y += 52) line(ctx, 0, y, width, y);
  ctx.restore();
}

function paintBackground(ctx, grid) {
  if (!grid?.points?.length) return;
  const cell = estimateCellSize(grid);
  grid.points.forEach((point) => {
    const screen = toScreen(point);
    ctx.save();
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = point.color;
    ctx.fillRect(screen.x - cell.width / 2, screen.y - cell.height / 2, cell.width + 1, cell.height + 1);
    ctx.restore();
  });
}

function estimateCellSize(grid) {
  const left = toScreen({ x: bounds.value.xMin, y: bounds.value.yMin });
  const right = toScreen({ x: bounds.value.xMax, y: bounds.value.yMin });
  const top = toScreen({ x: bounds.value.xMin, y: bounds.value.yMax });
  return {
    width: Math.abs(right.x - left.x) / Math.max(grid.columns - 1, 1),
    height: Math.abs(top.y - left.y) / Math.max(grid.rows - 1, 1),
  };
}

function paintHelpers(ctx, helpers) {
  helpers.forEach((helper) => {
    if (helper.type === "segment" || helper.type === "line") paintSegment(ctx, helper);
    if (helper.type === "point") paintHelperPoint(ctx, helper);
    if (helper.type === "rect") paintRect(ctx, helper);
    if (helper.type === "polyline") paintPolyline(ctx, helper);
  });
}

function paintSegment(ctx, helper) {
  const start = toScreen({ x: helper.x1, y: helper.y1 });
  const end = toScreen({ x: helper.x2, y: helper.y2 });
  ctx.save();
  ctx.strokeStyle = helper.color || "#0f172a";
  ctx.lineWidth = helper.width || 1.8;
  ctx.setLineDash(helper.dash || []);
  line(ctx, start.x, start.y, end.x, end.y);
  ctx.restore();
}

function paintHelperPoint(ctx, helper) {
  const screen = toScreen(helper);
  ctx.save();
  ctx.fillStyle = helper.color || "#0f172a";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(screen.x, screen.y, helper.radius || 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function paintRect(ctx, helper) {
  const topLeft = toScreen({ x: helper.xMin, y: helper.yMax });
  const bottomRight = toScreen({ x: helper.xMax, y: helper.yMin });
  ctx.save();
  ctx.fillStyle = helper.fill || "rgba(37,99,235,0.08)";
  ctx.strokeStyle = helper.color || "#2563eb";
  ctx.lineWidth = helper.width || 1;
  ctx.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
  ctx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
  ctx.restore();
}

function paintPolyline(ctx, helper) {
  if (!helper.points?.length) return;
  ctx.save();
  ctx.strokeStyle = helper.color || "#0f172a";
  ctx.lineWidth = helper.width || 2;
  ctx.setLineDash(helper.dash || []);
  helper.points.forEach((point, index) => {
    const screen = toScreen(point);
    if (index === 0) ctx.beginPath();
    if (index === 0) ctx.moveTo(screen.x, screen.y);
    else ctx.lineTo(screen.x, screen.y);
  });
  ctx.stroke();
  ctx.restore();
}

function paintScatter(ctx, points) {
  points.forEach((point) => {
    const screen = toScreen(point);
    ctx.save();
    ctx.globalAlpha = point.labelIndex === -1 ? 1 : 0.9;
    ctx.fillStyle = point.predictedColor || point.color;
    ctx.strokeStyle = point.selected ? "#111827" : point.misclassified ? "#f59e0b" : "#ffffff";
    ctx.lineWidth = point.selected || point.misclassified ? 2.8 : 1.5;
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, point.radius || 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (point.labelIndex === -1) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "700 12px sans-serif";
      ctx.fillText("?", screen.x - 3.5, screen.y + 4);
    }
    ctx.restore();
  });
}

function paintLabels(ctx, width, height) {
  const axis = coordinateSystem.value;
  ctx.save();
  ctx.fillStyle = "#475569";
  ctx.font = "12px sans-serif";
  ctx.fillText(axis.xFeatureName || "X", 54, height - 12);
  ctx.translate(14, height - 52);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(axis.yFeatureName || "Y", 0, 0);
  ctx.restore();
}

function paintStepBar(ctx, width, height) {
  const total = Math.max(steps.value.length, 1);
  const x = 18;
  const y = height - 20;
  const trackWidth = width - 36;
  ctx.save();
  ctx.fillStyle = "#dbe5f1";
  ctx.fillRect(x, y, trackWidth, 6);
  ctx.fillStyle = "#2563eb";
  ctx.fillRect(x, y, trackWidth * ((currentStep.value + 1) / total), 6);
  ctx.restore();
}

function toScreen(point) {
  const canvas = canvasRef.value;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const paddingX = 48;
  const top = 62;
  const bottom = 36;
  const xSpan = bounds.value.xMax - bounds.value.xMin || 1;
  const ySpan = bounds.value.yMax - bounds.value.yMin || 1;
  return {
    x: paddingX + ((point.x - bounds.value.xMin) / xSpan) * (width - paddingX * 2),
    y: height - bottom - ((point.y - bounds.value.yMin) / ySpan) * (height - top - bottom),
  };
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
</script>

<template>
  <div class="simulator step-simulator">
    <div class="simulator-controls frame-controls">
      <div class="frame-status">
        <span>Step {{ currentStep + 1 }} / {{ steps.length }}</span>
        <strong>{{ result.coordinateSystem?.xFeatureName }} × {{ result.coordinateSystem?.yFeatureName }}</strong>
      </div>
      <div class="legend-list">
        <span v-for="item in labels" :key="item.index">
          <i :style="{ background: item.color }"></i>
          {{ item.name }}
        </span>
      </div>
      <div class="simulator-actions">
        <button type="button" class="icon-text-button" :disabled="currentStep === 0" @click="previousStep">
          <ChevronLeft :size="17" />
          上一步
        </button>
        <button type="button" class="primary-action" :disabled="currentStep === steps.length - 1" @click="nextStep">
          下一步
          <ChevronRight :size="17" />
        </button>
        <button type="button" class="icon-text-button" @click="reset">
          <RotateCcw :size="17" />
          重置
        </button>
      </div>
    </div>

    <div class="simulator-body">
      <section ref="stageRef" class="simulator-stage">
        <canvas ref="canvasRef"></canvas>
      </section>
      <aside class="simulator-side">
        <div class="process-card">
          <span>状态帧 {{ currentStep + 1 }}</span>
          <h3>后端执行步骤</h3>
          <p>{{ frame.description }}</p>
        </div>

        <div v-if="frame.annotations?.length" class="annotation-panel">
          <span>后端计算备注</span>
          <p v-for="item in frame.annotations" :key="item.text" :class="item.tone">{{ item.text }}</p>
        </div>

        <div class="step-list-panel">
          <button
            v-for="item in steps"
            :key="item.index"
            type="button"
            :class="{ active: currentStep === item.index }"
            @click="currentStep = item.index"
          >
            Step {{ item.index + 1 }} · {{ item.description.slice(0, 28) }}
          </button>
        </div>
      </aside>
    </div>
  </div>
</template>
