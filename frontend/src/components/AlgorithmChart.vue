<script setup>
import InteractiveCanvas from "./InteractiveCanvas.vue";

defineProps({
  result: {
    type: Object,
    required: true,
  },
});
</script>

<template>
  <InteractiveCanvas :result="result" />
</template>
