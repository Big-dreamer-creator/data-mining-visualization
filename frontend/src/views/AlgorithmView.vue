<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { BookOpen, Database, RotateCw, SlidersHorizontal, X } from "lucide-vue-next";

import AlgorithmChart from "../components/AlgorithmChart.vue";
import DatasetPanel from "../components/DatasetPanel.vue";
import { fetchAlgorithm, fetchDataset, fetchDatasets, runAlgorithm } from "../services/api";
import { fallbackDatasets, tabs } from "../services/catalog";

const route = useRoute();
const activeTab = ref("principle");
const datasets = ref(fallbackDatasets);
const selectedDataset = ref("iris");
const datasetDetail = ref(null);
const algorithmInfo = ref(null);
const result = ref(null);
const loading = ref(false);
const error = ref("");
const showPrincipleDialog = ref(false);
const featureIndices = ref([2, 3]);
const hyperparameters = ref({});

const algorithmId = computed(() => route.params.id);
const algorithm = computed(() => algorithmInfo.value || { principle: [] });
const dataset = computed(() => result.value?.dataset || datasetDetail.value);
const featureOptions = computed(() => datasetDetail.value?.fields || dataset.value?.fields || []);
const parameterControls = computed(() => controlsFor(algorithmId.value));

function defaultFeatures(datasetId, detail) {
  const fields = detail?.fields || [];
  const candidate = datasetId === "wine" ? [6, 12] : [2, 3];
  if (candidate.every((index) => index < fields.length)) return candidate;
  return [0, Math.min(1, Math.max(fields.length - 1, 0))];
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
  try {
    algorithmInfo.value = await fetchAlgorithm(algorithmId.value);
  } catch (apiError) {
    algorithmInfo.value = null;
  }
}

async function loadDatasetDetail() {
  datasetDetail.value = await fetchDataset(selectedDataset.value);
  featureIndices.value = defaultFeatures(selectedDataset.value, datasetDetail.value);
}

async function loadRun() {
  loading.value = true;
  error.value = "";
  try {
    result.value = await runAlgorithm(algorithmId.value, {
      datasetId: selectedDataset.value,
      featureIndices: featureIndices.value.map((item) => Number(item)),
      hyperparameters: { ...hyperparameters.value },
    });
  } catch (apiError) {
    error.value = "后端接口暂不可用，或当前参数组合无法生成执行过程。";
    result.value = null;
  } finally {
    loading.value = false;
  }
}

async function reloadAll({ resetParameters = false, resetFeatures = false } = {}) {
  if (resetParameters) hyperparameters.value = defaultHyperparameters(algorithmId.value);
  if (resetFeatures || !datasetDetail.value) await loadDatasetDetail();
  await loadAlgorithmInfo();
  await loadRun();
}

function updateFeature(axis, value) {
  const next = [...featureIndices.value];
  next[axis] = Number(value);
  if (next[0] === next[1]) return;
  featureIndices.value = next;
}

onMounted(async () => {
  hyperparameters.value = defaultHyperparameters(algorithmId.value);
  try {
    datasets.value = await fetchDatasets();
  } catch (apiError) {
    datasets.value = fallbackDatasets;
  }
  await reloadAll({ resetFeatures: true });
});

watch(
  () => algorithmId.value,
  async () => {
    activeTab.value = "principle";
    await reloadAll({ resetParameters: true });
  },
);

watch(
  () => selectedDataset.value,
  async () => {
    await reloadAll({ resetFeatures: true });
  },
);
</script>

<template>
  <section class="algorithm-page">
    <div class="toolbar">
      <div class="dataset-select">
        <Database :size="18" />
        <select v-model="selectedDataset" aria-label="选择数据集">
          <option v-for="item in datasets" :key="item.id" :value="item.id">
            {{ item.name }}
          </option>
        </select>
      </div>
      <div class="tab-list" role="tablist">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          class="tab-button"
          :class="{ active: activeTab === tab.id }"
          type="button"
          @click="activeTab = tab.id"
        >
          {{ tab.label }}
        </button>
      </div>
    </div>

    <div v-if="loading" class="state-panel">
      <RotateCw :size="24" class="spin" />
      正在由后端生成完整执行过程状态帧
    </div>
    <div v-else-if="error" class="state-panel error">{{ error }}</div>

    <template v-else-if="result">
      <section v-if="activeTab === 'principle'" class="content-panel">
        <div class="section-heading">
          <div>
            <span class="eyebrow">{{ algorithm.category }}</span>
            <h2>{{ algorithm.name || result.algorithmName }}</h2>
          </div>
          <button class="icon-text-button" type="button" @click="showPrincipleDialog = true">
            <BookOpen :size="18" />
            深入说明
          </button>
        </div>
        <div class="principle-grid">
          <article v-for="item in algorithm.principle" :key="item.title" class="principle-card">
            <span>{{ item.title }}</span>
            <p>{{ item.body }}</p>
          </article>
        </div>
      </section>

      <DatasetPanel v-if="activeTab === 'dataset' && dataset" :dataset="dataset" />

      <section v-if="activeTab === 'visualization'" class="content-panel chart-panel">
        <div class="section-heading">
          <div>
            <span class="eyebrow">Process Frame Player</span>
            <h2>算法执行过程播放器</h2>
          </div>
        </div>
        <div class="inline-controls">
          <div class="inline-controls-head">
            <div>
              <span>执行过程输入</span>
              <strong>后端按当前输入重新生成完整状态帧</strong>
            </div>
            <button class="primary-action" type="button" :disabled="loading" @click="loadRun">
              <SlidersHorizontal :size="17" />
              重新计算
            </button>
          </div>
          <div class="control-grid compact-controls">
            <label>
              <span>X 特征</span>
              <select :value="featureIndices[0]" @change="updateFeature(0, $event.target.value)">
                <option v-for="(field, index) in featureOptions" :key="field.name" :value="index" :disabled="index === featureIndices[1]">
                  {{ field.name }}
                </option>
              </select>
            </label>
            <label>
              <span>Y 特征</span>
              <select :value="featureIndices[1]" @change="updateFeature(1, $event.target.value)">
                <option v-for="(field, index) in featureOptions" :key="field.name" :value="index" :disabled="index === featureIndices[0]">
                  {{ field.name }}
                </option>
              </select>
            </label>
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
        <AlgorithmChart :result="result" />
      </section>

      <section v-if="activeTab === 'analysis'" class="content-panel">
        <div class="section-heading">
          <div>
            <span class="eyebrow">Result Analysis</span>
            <h2>结果分析</h2>
          </div>
        </div>
        <div class="analysis-grid">
          <article v-for="section in result.analysis" :key="section.title" class="analysis-card">
            <h3>{{ section.title }}</h3>
            <p>{{ section.body }}</p>
          </article>
        </div>
      </section>
    </template>

    <div v-if="showPrincipleDialog && algorithm" class="dialog-backdrop" @click.self="showPrincipleDialog = false">
      <div class="dialog">
        <button class="dialog-close" type="button" aria-label="关闭" @click="showPrincipleDialog = false">
          <X :size="20" />
        </button>
        <span class="eyebrow">Algorithm Principle</span>
        <h2>{{ algorithm.name || result?.algorithmName }}</h2>
        <div class="principle-grid detailed">
          <article v-for="item in algorithm.principle" :key="item.title" class="principle-card">
            <span>{{ item.title }}</span>
            <p>{{ item.body }}</p>
          </article>
          <article class="principle-card">
            <span>页面职责</span>
            <p>
              可视化只负责播放后端返回的算法执行状态帧；距离、投影、边界、切分、权重和网格预测均由后端提前计算。
            </p>
          </article>
        </div>
      </div>
    </div>
  </section>
</template>
