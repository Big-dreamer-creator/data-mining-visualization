import { createApp } from "vue";
import { createRouter, createWebHistory } from "vue-router";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { BarChart, GraphChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import VChart from "vue-echarts";

import App from "./App.vue";
import HomeView from "./views/HomeView.vue";
import AlgorithmView from "./views/AlgorithmView.vue";
import "./styles/global.css";

use([
  CanvasRenderer,
  BarChart,
  GraphChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
]);

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "home", component: HomeView },
    { path: "/algorithm/:id", name: "algorithm", component: AlgorithmView },
  ],
});

createApp(App).component("VChart", VChart).use(router).mount("#app");
