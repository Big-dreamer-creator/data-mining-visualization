<script setup>
import { onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { Activity, Database, PlayCircle, Target } from "lucide-vue-next";

import { fetchAlgorithms } from "../services/api";
import { fallbackAlgorithms } from "../services/catalog";

const algorithms = ref(fallbackAlgorithms);

onMounted(async () => {
  try {
    algorithms.value = await fetchAlgorithms();
  } catch (error) {
    algorithms.value = fallbackAlgorithms;
  }
});
</script>

<template>
  <section class="home-grid">
    <div class="intro-panel">
      <span class="eyebrow">Vue 3 + FastAPI + ECharts</span>
      <h2>面向教学演示的数据挖掘分类算法可视化系统</h2>
      <p>
        系统围绕六类经典分类算法组织页面：独立可视化模块用于演示算法本身的执行过程，真实数据评价模块用于展示预测表现和评价指标。
      </p>
      <div class="workflow">
        <div><PlayCircle :size="20" /> 独立过程演示</div>
        <div><Activity :size="20" /> 状态帧播放</div>
        <div><Database :size="20" /> 真实数据评测</div>
        <div><Target :size="20" /> 指标与预测表现</div>
      </div>
    </div>

    <div class="algorithm-grid">
      <RouterLink
        v-for="algorithm in algorithms"
        :key="algorithm.id"
        class="algorithm-card"
        :to="`/algorithm/${algorithm.id}`"
      >
        <span>{{ algorithm.shortName }}</span>
        <h3>{{ algorithm.name }}</h3>
        <p>{{ algorithm.summary }}</p>
      </RouterLink>
    </div>
  </section>
</template>
