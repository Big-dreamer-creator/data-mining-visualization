<script setup>
import { computed } from "vue";

const props = defineProps({
  result: {
    type: Object,
    required: true,
  },
});

const matrixOption = computed(() => {
  const labels = props.result.charts.confusionMatrix.labels;
  const values = props.result.charts.confusionMatrix.values.flatMap((row, y) =>
    row.map((value, x) => [x, y, value]),
  );

  return {
    tooltip: { position: "top" },
    grid: { left: 90, right: 30, top: 30, bottom: 70 },
    xAxis: { type: "category", data: labels, name: "预测类别", axisLabel: { color: "#526070" } },
    yAxis: { type: "category", data: labels, name: "真实类别", axisLabel: { color: "#526070" } },
    visualMap: {
      min: 0,
      max: Math.max(...values.map((item) => item[2]), 1),
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 4,
      inRange: { color: ["#e8f1ff", "#2563eb"] },
    },
    series: [
      {
        type: "heatmap",
        data: values,
        label: { show: true, color: "#0f172a" },
      },
    ],
  };
});

function formatMetric(metric) {
  if (metric.format === "percent") return `${(metric.value * 100).toFixed(2)}%`;
  return metric.value;
}
</script>

<template>
  <section class="content-panel">
    <div class="section-heading">
      <div>
        <span class="eyebrow">Evaluation</span>
        <h2>结果评测</h2>
      </div>
    </div>

    <div class="metric-grid">
      <div v-for="metric in result.metrics" :key="metric.label" class="metric-card">
        <span>{{ metric.label }}</span>
        <strong>{{ formatMetric(metric) }}</strong>
      </div>
    </div>

    <div class="sub-panel">
      <h3>混淆矩阵</h3>
      <VChart class="main-chart" :option="matrixOption" autoresize />
    </div>

    <div class="sub-panel table-panel">
      <h3>逐类别指标</h3>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>类别</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in result.charts.classMetrics" :key="item.label">
              <td>{{ item.label }}</td>
              <td>{{ (item.precision * 100).toFixed(2) }}%</td>
              <td>{{ (item.recall * 100).toFixed(2) }}%</td>
              <td>{{ (item.f1 * 100).toFixed(2) }}%</td>
              <td>{{ item.support }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
