<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { BookOpen, Database, Orbit, RotateCw, SlidersHorizontal, X } from "lucide-vue-next";

import AlgorithmChart from "../components/AlgorithmChart.vue";
import DatasetPanel from "../components/DatasetPanel.vue";
import MetricsPanel from "../components/MetricsPanel.vue";
import { evaluateAlgorithm, fetchAlgorithm, fetchDatasets, runVisualization } from "../services/api";
import { fallbackAlgorithms, fallbackDatasets } from "../services/catalog";

const route = useRoute();
const activeTab = ref("principle");

// 直接在前端定义独立展示用的数据分布模式
const patterns = [
  { id: "blobs", name: "分布：线性可分 (Blobs)" },
  { id: "moons", name: "分布：半月形非线性 (Moons)" },
  { id: "circles", name: "分布：环形包裹非线性 (Circles)" }
];
const selectedPattern = ref("blobs");
const datasets = ref(fallbackDatasets);
const selectedDataset = ref("iris");

const algorithmInfo = ref(null);
const visualizationResult = ref(null);
const evaluationResult = ref(null);
const loading = ref(false);
const error = ref("");
const principleError = ref("");
const visualizationError = ref("");
const evaluationError = ref("");
const showPrincipleDialog = ref(false);
const hyperparameters = ref({});

const algorithmId = computed(() => route.params.id);
const fallbackAlgorithm = computed(() => fallbackAlgorithms.find((item) => item.id === algorithmId.value));
const algorithm = computed(() => mergeAlgorithmInfo(algorithmInfo.value, fallbackAlgorithm.value));
const parameterControls = computed(() => controlsFor(algorithmId.value));
const dataset = computed(() => evaluationResult.value?.dataset);

function mergeAlgorithmInfo(remote, fallback) {
  if (!remote && !fallback) return { principle: [], deepDive: [] };
  const local = fallback || {};
  const api = remote || {};
  const apiPrinciple = Array.isArray(api.principle) ? api.principle : [];
  const localPrinciple = Array.isArray(local.principle) ? local.principle : [];
  const apiDeepDive = Array.isArray(api.deepDive) ? api.deepDive : [];
  const localDeepDive = Array.isArray(local.deepDive) ? local.deepDive : [];
  return {
    ...local,
    ...api,
    principle: apiPrinciple.length >= 5 ? apiPrinciple : localPrinciple,
    deepDive: apiDeepDive.length >= 4 ? apiDeepDive : localDeepDive,
  };
}

function defaultHyperparameters(id) {
  const defaults = {
    knn: { k: 5, distanceMetric: "euclidean" },
    "logistic-regression": { learningRate: 0.12, iterations: 260, l2: 0.001, randomSeed: 7 },
    lda: { regularization: 0.001, thresholdMode: "nearest-center" },
    "decision-tree": { criterion: "gini", maxDepth: 4, minSamplesLeaf: 3 },
    adaboost: { nEstimators: 5, learningRate: 0.8 },
    mlp: { hiddenLayers: "32,16", activation: "tanh", learningRate: 0.025, alpha: 0.0003, epochs: 140, randomSeed: 42 },
  };
  return { ...(defaults[id] || {}) };
}

function controlsFor(id) {
  const common = {
    knn: [
      { key: "k", label: "K 值", type: "number", min: 1, max: 35, step: 1 },
      { key: "distanceMetric", label: "距离度量", type: "select", options: [["euclidean", "欧氏距离"], ["manhattan", "曼哈顿距离"]] },
    ],
    "logistic-regression": [
      { key: "learningRate", label: "学习率", type: "number", min: 0.001, max: 2, step: 0.001 },
      { key: "iterations", label: "迭代次数", type: "number", min: 30, max: 1000, step: 10 },
      { key: "l2", label: "L2 正则", type: "number", min: 0, max: 2, step: 0.001 },
      { key: "randomSeed", label: "随机种子", type: "number", min: 0, max: 100000, step: 1 },
    ],
    lda: [
      { key: "regularization", label: "正则项", type: "number", min: 0, max: 1, step: 0.001 },
      { key: "thresholdMode", label: "阈值模式", type: "select", options: [["nearest-center", "最近中心"]] },
    ],
    "decision-tree": [
      { key: "criterion", label: "切分准则", type: "select", options: [["gini", "Gini"], ["entropy", "Entropy"], ["log_loss", "Log Loss"]] },
      { key: "maxDepth", label: "最大深度", type: "number", min: 1, max: 8, step: 1 },
      { key: "minSamplesLeaf", label: "叶子最小样本", type: "number", min: 1, max: 30, step: 1 },
    ],
    adaboost: [
      { key: "nEstimators", label: "弱分类器数", type: "number", min: 1, max: 12, step: 1 },
      { key: "learningRate", label: "学习率", type: "number", min: 0.05, max: 3, step: 0.05 },
    ],
    mlp: [
      { key: "hiddenLayers", label: "隐藏层", type: "text" },
      { key: "activation", label: "激活函数", type: "select", options: [["tanh", "Tanh"], ["relu", "ReLU"], ["logistic", "Logistic"], ["identity", "Identity"]] },
      { key: "learningRate", label: "学习率", type: "number", min: 0.0001, max: 1, step: 0.0001 },
      { key: "alpha", label: "Alpha", type: "number", min: 0, max: 1, step: 0.0001 },
      { key: "epochs", label: "Epochs", type: "number", min: 10, max: 400, step: 10 },
      { key: "randomSeed", label: "随机种子", type: "number", min: 0, max: 100000, step: 1 },
    ],
  };
  return common[id] || [];
}

async function loadAlgorithmInfo() {
  principleError.value = "";
  try {
    algorithmInfo.value = await fetchAlgorithm(algorithmId.value);
  } catch (apiError) {
    principleError.value = "算法原理暂未加载，请确认后端算法接口可用。";
    algorithmInfo.value = null;
  }
}

async function loadVisualization(options = {}) {
  const manageLoading = options.manageLoading !== false;
  if (manageLoading) loading.value = true;
  visualizationError.value = "";
  try {
    visualizationResult.value = await runVisualization(algorithmId.value, {
      pattern: selectedPattern.value,
      hyperparameters: { ...hyperparameters.value },
    });
  } catch (apiError) {
    visualizationError.value = "独立可视化数据暂未加载，请确认后端可视化接口可用。";
    visualizationResult.value = null;
  } finally {
    if (manageLoading) loading.value = false;
  }
}

async function loadEvaluation(options = {}) {
  const manageLoading = options.manageLoading !== false;
  if (manageLoading) loading.value = true;
  evaluationError.value = "";
  try {
    evaluationResult.value = await evaluateAlgorithm(algorithmId.value, {
      datasetId: selectedDataset.value,
      hyperparameters: { ...hyperparameters.value },
    });
  } catch (apiError) {
    evaluationError.value = "真实数据评价暂未加载，请确认后端评价接口可用。";
    evaluationResult.value = null;
  } finally {
    if (manageLoading) loading.value = false;
  }
}

async function reloadAll({ resetParameters = false } = {}) {
  if (resetParameters) hyperparameters.value = defaultHyperparameters(algorithmId.value);
  loading.value = true;
  error.value = "";
  await loadAlgorithmInfo();
  await Promise.allSettled([
    loadVisualization({ manageLoading: false }),
    loadEvaluation({ manageLoading: false }),
  ]);
  loading.value = false;
}

async function recomputeWithCurrentParameters() {
  loading.value = true;
  await Promise.allSettled([
    loadVisualization({ manageLoading: false }),
    loadEvaluation({ manageLoading: false }),
  ]);
  loading.value = false;
}

onMounted(async () => {
  hyperparameters.value = defaultHyperparameters(algorithmId.value);
  try {
    datasets.value = await fetchDatasets();
  } catch (apiError) {
    datasets.value = fallbackDatasets;
  }
  await reloadAll();
});

watch(
  () => algorithmId.value,
  async () => {
    activeTab.value = "principle";
    await reloadAll({ resetParameters: true });
  },
);

watch(
  () => selectedPattern.value,
  async () => {
    await loadVisualization();
  },
);

watch(
  () => selectedDataset.value,
  async () => {
    await loadEvaluation();
  },
);
</script>

<template>
  <section class="algorithm-page">
    <div class="toolbar">
      <div class="dataset-select">
        <Orbit :size="18" />
        <select v-model="selectedPattern" aria-label="选择数据模式">
          <option v-for="item in patterns" :key="item.id" :value="item.id">
            {{ item.name }}
          </option>
        </select>
      </div>
      <div class="dataset-select">
        <Database :size="18" />
        <select v-model="selectedDataset" aria-label="选择评价数据集">
          <option v-for="item in datasets" :key="item.id" :value="item.id">
            {{ item.name }}
          </option>
        </select>
      </div>
      <div class="tab-list" role="tablist">
        <button
          class="tab-button"
          :class="{ active: activeTab === 'principle' }"
          type="button"
          @click="activeTab = 'principle'"
        >
          原理讲解
        </button>
        <button
          class="tab-button"
          :class="{ active: activeTab === 'visualization' }"
          type="button"
          @click="activeTab = 'visualization'"
        >
          独立可视化运行
        </button>
        <button
          class="tab-button"
          :class="{ active: activeTab === 'dataset' }"
          type="button"
          @click="activeTab = 'dataset'"
        >
          数据集
        </button>
        <button
          class="tab-button"
          :class="{ active: activeTab === 'metrics' }"
          type="button"
          @click="activeTab = 'metrics'"
        >
          结果评价
        </button>
        <button
          class="tab-button"
          :class="{ active: activeTab === 'analysis' }"
          type="button"
          @click="activeTab = 'analysis'"
        >
          结果分析
        </button>
      </div>
    </div>

    <div v-if="loading" class="state-panel">
      <RotateCw :size="24" class="spin" />
      正在由后端计算数据
    </div>
    <div v-else-if="error" class="state-panel error">{{ error }}</div>

    <template v-else>
      <section v-if="activeTab === 'principle'" class="content-panel">
        <div class="section-heading">
          <div>
            <span class="eyebrow">{{ algorithm.category }}</span>
            <h2>{{ algorithm.name || visualizationResult?.algorithmName }}</h2>
          </div>
          <button class="icon-text-button" type="button" @click="showPrincipleDialog = true">
            <BookOpen :size="18" />
            深入说明
          </button>
        </div>
        <div v-if="principleError" class="inline-state error">{{ principleError }}</div>
        <div class="principle-grid">
          <article v-for="item in algorithm.principle" :key="item.title" class="principle-card">
            <span>{{ item.title }}</span>
            <p>{{ item.body }}</p>
          </article>
        </div>
        <div v-if="!principleError && !algorithm.principle.length" class="inline-state">
          暂无原理内容。
        </div>
      </section>

      <DatasetPanel v-if="activeTab === 'dataset' && dataset" :dataset="dataset" />
      <section v-else-if="activeTab === 'dataset'" class="content-panel">
        <div class="inline-state" :class="{ error: evaluationError }">
          {{ evaluationError || "数据集信息正在等待评价接口返回。" }}
        </div>
      </section>

      <section v-if="activeTab === 'visualization' && visualizationResult" class="content-panel chart-panel">
        <div class="section-heading">
          <div>
            <span class="eyebrow">Standalone Algorithm Demo</span>
            <h2>纯净算法过程演示</h2>
          </div>
        </div>
        <div class="inline-controls">
          <div class="inline-controls-head">
            <div>
              <span>超参数与模型输入</span>
              <strong>调整参数观察模型如何在二维平面改变几何边界</strong>
            </div>
            <button class="primary-action" type="button" :disabled="loading" @click="recomputeWithCurrentParameters">
              <SlidersHorizontal :size="17" />
              重新计算
            </button>
          </div>
          <div class="control-grid compact-controls">
            <label v-for="control in parameterControls" :key="control.key">
              <span>{{ control.label }}</span>
              <select v-if="control.type === 'select'" v-model="hyperparameters[control.key]">
                <option v-for="[value, label] in control.options" :key="value" :value="value">{{ label }}</option>
              </select>
              <input
                v-else-if="control.type === 'number'"
                v-model.number="hyperparameters[control.key]"
                type="number"
                :min="control.min"
                :max="control.max"
                :step="control.step"
              />
              <input v-else v-model="hyperparameters[control.key]" type="text" />
            </label>
          </div>
        </div>
        <AlgorithmChart :result="visualizationResult" />
      </section>
      <section v-else-if="activeTab === 'visualization'" class="content-panel">
        <div class="inline-state" :class="{ error: visualizationError }">
          {{ visualizationError || "可视化状态帧正在等待加载。" }}
        </div>
      </section>

      <MetricsPanel v-if="activeTab === 'metrics' && evaluationResult" :result="evaluationResult" />
      <section v-else-if="activeTab === 'metrics'" class="content-panel">
        <div class="inline-state" :class="{ error: evaluationError }">
          {{ evaluationError || "真实数据评价正在等待加载。" }}
        </div>
      </section>

      <section v-if="activeTab === 'analysis' && evaluationResult" class="content-panel">
        <div class="section-heading">
          <div>
            <span class="eyebrow">Result Analysis</span>
            <h2>真实数据结果分析</h2>
          </div>
        </div>
        <div class="analysis-grid">
          <article v-for="section in evaluationResult.analysis" :key="section.title" class="analysis-card">
            <h3>{{ section.title }}</h3>
            <p>{{ section.body }}</p>
          </article>
        </div>
      </section>
      <section v-else-if="activeTab === 'analysis'" class="content-panel">
        <div class="inline-state" :class="{ error: evaluationError }">
          {{ evaluationError || "结果分析正在等待评价数据。" }}
        </div>
      </section>
    </template>

    <div v-if="showPrincipleDialog && algorithm" class="dialog-backdrop" @click.self="showPrincipleDialog = false">
      <div class="dialog">
        <button class="dialog-close" type="button" aria-label="关闭" @click="showPrincipleDialog = false">
          <X :size="20" />
        </button>
        <span class="eyebrow">Algorithm Principle</span>
        <h2>{{ algorithm.name || visualizationResult?.algorithmName }}</h2>
        <div class="principle-grid detailed">
          <article v-for="item in algorithm.principle" :key="item.title" class="principle-card">
            <span>{{ item.title }}</span>
            <p>{{ item.body }}</p>
          </article>
        </div>
        <div v-if="algorithm.deepDive?.length" class="dialog-section-title">
          深入理解
        </div>
        <div v-if="algorithm.deepDive?.length" class="principle-grid detailed">
          <article v-for="item in algorithm.deepDive" :key="item.title" class="principle-card">
            <span>{{ item.title }}</span>
            <p>{{ item.body }}</p>
          </article>
          <article class="principle-card">
            <span>模块架构说明</span>
            <p>
              独立可视化模块使用二维合成二分类数据解释算法过程；结果评价模块使用真实数据集和全部特征评估预测表现，两者互不干扰。
            </p>
          </article>
        </div>
      </div>
    </div>
  </section>
</template>
