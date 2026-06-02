export const MODEL_COLORS: Record<string, string> = {
  "naive-seasonal": "#E4E4E7",
  "sarima":         "#06B6D4",
  "ridge-exog":     "#F59E0B",
  "timesfm":        "#8B5CF6",
  "chronos-2":      "#10B981",
  "timegpt":        "#F43F5E",
};

const FALLBACKS = ["#8B5CF6", "#06B6D4", "#F59E0B", "#10B981", "#F43F5E", "#E4E4E7"];

export function modelColor(slug: string, index = 0): string {
  return MODEL_COLORS[slug] ?? FALLBACKS[index % FALLBACKS.length];
}
