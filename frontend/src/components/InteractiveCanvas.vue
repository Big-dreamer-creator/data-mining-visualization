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

// 全局统一缩放比例与偏移量，保证几何等宽等高，不破坏空间距离度量和正交性
const scale = ref(1);
const offsetX = ref(0);
const offsetY = ref(0);

watch(
  () => props.result,
  () => reset(),
  { deep: true },
);

watch(currentStep, () => {
  requestAnimationFrame(draw); // 使用 RAF 保证滑块拖动或播放时的丝滑渲染
});

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

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  // 计算等比例坐标系转换参数 (Isotropic Scaling)
  const paddingX = 48;
  const top = 62;
  const bottom = 48;
  const xSpan = bounds.value.xMax - bounds.value.xMin || 1;
  const ySpan = bounds.value.yMax - bounds.value.yMin || 1;

  const drawWidth = rect.width - paddingX * 2;
  const drawHeight = height - top - bottom;

  // 取 X 和 Y 中最小的缩放比，保证数学上的绝对比例正确
  scale.value = Math.min(drawWidth / xSpan, drawHeight / ySpan);

  // 计算居中偏移量
  offsetX.value = paddingX + (drawWidth - xSpan * scale.value) / 2;
  offsetY.value = height - bottom - (drawHeight - ySpan * scale.value) / 2;

  draw();
}

function draw() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = parseFloat(canvas.style.width);
  const height = parseFloat(canvas.style.height);

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
  ctx.fillText(`${props.result.algorithmName || 'Algorithm'} · 后端状态帧播放`, 18, 27);
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
  ctx.save();

  const groups = {};
  grid.points.forEach((point) => {
    // 映射后端传来的 alpha（置信度），如果未传则默认使用 0.16 的低透明度
    const alpha = point.alpha !== undefined ? (point.alpha * 0.55).toFixed(2) : 0.16;
    const key = `${point.color}-${alpha}`;
    if (!groups[key]) groups[key] = { color: point.color, alpha: alpha, points: [] };
    groups[key].points.push(point);
  });

  const cellW = ((bounds.value.xMax - bounds.value.xMin) / Math.max(grid.columns - 1, 1)) * scale.value;
  const cellH = ((bounds.value.yMax - bounds.value.yMin) / Math.max(grid.rows - 1, 1)) * scale.value;
  // 增加少许像素重叠（1.5px），消除单元格之间的白边抗锯齿缝隙
  const renderW = cellW + 1.5;
  const renderH = cellH + 1.5;

  for (const key in groups) {
    const group = groups[key];
    ctx.fillStyle = group.color;
    ctx.globalAlpha = parseFloat(group.alpha);
    ctx.beginPath();
    for (let i = 0; i < group.points.length; i++) {
      const screen = toScreen(group.points[i]);
      ctx.rect(screen.x - cellW / 2, screen.y - cellH / 2, renderW, renderH);
    }
    ctx.fill(); // 一次性填充同一颜色和透明度的所有网格点，大幅提升渲染性能
  }
  ctx.restore();
}

function paintHelpers(ctx, helpers) {
  helpers.forEach((helper) => {
    if (helper.type === "segment" || helper.type === "line") paintSegment(ctx, helper);
    if (helper.type === "point") paintHelperPoint(ctx, helper);
    if (helper.type === "rect") paintRect(ctx, helper);
    if (helper.type === "polyline") paintPolyline(ctx, helper);
    if (helper.type === "circle") paintCircle(ctx, helper);
  });
}

function paintCircle(ctx, helper) {
  const center = toScreen({ x: helper.x, y: helper.y });
  const radiusPx = helper.r * scale.value; // 将特征空间的距离转换为屏幕像素
  ctx.save();
  ctx.strokeStyle = helper.color || "#0f172a";
  ctx.lineWidth = helper.width || 1.5;
  if (helper.dash?.length) ctx.setLineDash(helper.dash);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radiusPx, 0, Math.PI * 2);
  ctx.stroke();
  if (helper.fill) {
    ctx.fillStyle = helper.fill;
    ctx.fill();
  }
  ctx.restore();
}

function paintSegment(ctx, helper) {
  const start = toScreen({ x: helper.x1, y: helper.y1 });
  const end = toScreen({ x: helper.x2, y: helper.y2 });
  ctx.save();
  ctx.strokeStyle = helper.color || "#0f172a";
  ctx.lineWidth = helper.width || 1.8;
  if (helper.dash?.length) ctx.setLineDash(helper.dash);
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
  if (helper.dash?.length) ctx.setLineDash(helper.dash);
  ctx.beginPath();
  helper.points.forEach((point, index) => {
    const screen = toScreen(point);
    if (index === 0) ctx.moveTo(screen.x, screen.y);
    else ctx.lineTo(screen.x, screen.y);
  });
  ctx.stroke();
  ctx.restore();
}

function paintScatter(ctx, points) {
  // 核心逻辑修正：小散点和被错分的重点样本画在最上层，防止被 AdaBoost 等大权重点吞没
  const sortedPoints = [...points].sort((a, b) => {
    if (a.misclassified !== b.misclassified) return a.misclassified ? 1 : -1;
    return (b.radius || 6) - (a.radius || 6);
  });

  sortedPoints.forEach((point) => {
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
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("?", screen.x, screen.y + 1); // 居中问号
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
  const progress = total === 1 ? 1 : (currentStep.value + 1) / total;
  ctx.fillRect(x, y, trackWidth * progress, 6);
  ctx.restore();
}

// 基于统一缩放率计算屏幕坐标系
function toScreen(point) {
  return {
    x: offsetX.value + (point.x - bounds.value.xMin) * scale.value,
    y: offsetY.value - (point.y - bounds.value.yMin) * scale.value,
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