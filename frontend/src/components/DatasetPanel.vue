<script setup>
import { computed } from "vue";

const props = defineProps({
  dataset: {
    type: Object,
    required: true,
  },
});

const distributionOption = computed(() => ({
  tooltip: { trigger: "axis" },
  grid: { left: 42, right: 18, top: 28, bottom: 36 },
  xAxis: {
    type: "category",
    data: props.dataset.targetDistribution.map((item) => item.label),
    axisLabel: { color: "#526070" },
  },
  yAxis: { type: "value", axisLabel: { color: "#526070" } },
  series: [
    {
      type: "bar",
      data: props.dataset.targetDistribution.map((item) => item.count),
      itemStyle: { color: "#2563eb", borderRadius: [4, 4, 0, 0] },
    },
  ],
}));
</script>

<template>
  <section class="content-panel dataset-panel">
    <div class="section-heading">
      <div>
        <span class="eyebrow">{{ dataset.task }}</span>
        <h2>{{ dataset.name }}</h2>
      </div>
      <span class="dataset-file">{{ dataset.file }}</span>
    </div>

    <p class="muted">{{ dataset.description }}</p>
    <p class="source">公开来源：{{ dataset.source }}</p>

    <div class="stat-strip">
      <div>
        <span>样本数</span>
        <strong>{{ dataset.sampleCount }}</strong>
      </div>
      <div>
        <span>特征数</span>
        <strong>{{ dataset.featureCount }}</strong>
      </div>
      <div>
        <span>目标字段</span>
        <strong>{{ dataset.targetLabel }}</strong>
      </div>
    </div>

    <div class="two-column">
      <div class="sub-panel">
        <h3>类别分布</h3>
        <VChart class="small-chart" :option="distributionOption" autoresize />
      </div>
      <div class="sub-panel">
        <h3>字段概览</h3>
        <div class="field-list">
          <div v-for="field in dataset.fields.slice(0, 10)" :key="field.name" class="field-row">
            <span>{{ field.name }}</span>
            <small>mean {{ field.mean }} / std {{ field.std }}</small>
          </div>
        </div>
      </div>
    </div>

    <div class="sub-panel table-panel">
      <h3>数据局部预览</h3>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th v-for="field in Object.keys(dataset.preview?.[0] || {})" :key="field">{{ field }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in dataset.preview" :key="rowIndex">
              <td v-for="field in Object.keys(row)" :key="field">{{ row[field] }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
