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
        系统围绕六类经典分类算法组织页面，将算法原理、真实数据集和后端生成的执行过程状态帧整合到同一工作区。
      </p>
      <div class="workflow">
        <div><Database :size="20" /> 数据集选择</div>
        <div><PlayCircle :size="20" /> 模型运行</div>
        <div><Activity :size="20" /> 状态帧播放</div>
        <div><Target :size="20" /> 过程观察</div>
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
