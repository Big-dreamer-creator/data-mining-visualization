<script setup>
import { computed, onMounted, ref } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";
import { BarChart3, BrainCircuit, GitFork, Home, Network, Route, Sigma, Waypoints } from "lucide-vue-next";

import { fetchAlgorithms } from "./services/api";
import { fallbackAlgorithms } from "./services/catalog";

const route = useRoute();
const algorithms = ref(fallbackAlgorithms);

const iconMap = {
  knn: Waypoints,
  "logistic-regression": Sigma,
  lda: Route,
  "decision-tree": GitFork,
  adaboost: BarChart3,
  mlp: Network,
};

const currentTitle = computed(() => {
  if (route.name === "home") return "首页";
  return algorithms.value.find((item) => item.id === route.params.id)?.name || "算法详情";
});

onMounted(async () => {
  try {
    algorithms.value = await fetchAlgorithms();
  } catch (error) {
    algorithms.value = fallbackAlgorithms;
  }
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">
          <BrainCircuit :size="24" />
        </div>
        <div>
          <strong>分类算法可视化</strong>
          <span>Data Mining Lab</span>
        </div>
      </div>

      <nav class="nav-list" aria-label="主导航">
        <RouterLink class="nav-item" to="/">
          <Home :size="18" />
          <span>首页</span>
        </RouterLink>
        <RouterLink
          v-for="algorithm in algorithms"
          :key="algorithm.id"
          class="nav-item"
          :to="`/algorithm/${algorithm.id}`"
        >
          <component :is="iconMap[algorithm.id] || BarChart3" :size="18" />
          <span>{{ algorithm.name }}</span>
        </RouterLink>
      </nav>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <span class="eyebrow">分类算法教学 Demo</span>
          <h1>{{ currentTitle }}</h1>
        </div>
      </header>
      <RouterView />
    </main>
  </div>
</template>
