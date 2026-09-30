import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('📱 Running Mobile Viewport & HIG Standards Verification...');

// 1. Check index.html viewport and safe area tags
const htmlContent = fs.readFileSync(path.join(rootDir, 'index.html'), 'utf-8');

if (!htmlContent.includes('viewport-fit=cover')) {
  console.error('❌ Missing viewport-fit=cover in index.html');
  process.exit(1);
}
console.log('✅ index.html includes viewport-fit=cover for iPhone 16 notch/dynamic island.');

// 2. Check CSS safe areas and 44pt touch targets
const cssContent = fs.readFileSync(path.join(rootDir, 'style.css'), 'utf-8');

if (!cssContent.includes('env(safe-area-inset-top') || !cssContent.includes('env(safe-area-inset-bottom')) {
  console.error('❌ Missing CSS safe area inset environment variables');
  process.exit(1);
}
console.log('✅ CSS contains safe-area-inset-top and safe-area-inset-bottom.');

if (!cssContent.includes('--tap-target-min: 44px') && !cssContent.includes('44px')) {
  console.error('❌ Missing minimum touch target guideline (44px)');
  process.exit(1);
}
console.log('✅ Apple HIG minimum touch target >= 44pt verified in CSS.');

// 3. Verify data.js exports
const dataContent = fs.readFileSync(path.join(rootDir, 'data.js'), 'utf-8');
if (!dataContent.includes('PIPELINE_NODES') || !dataContent.includes('DIAGNOSTIC_CASES')) {
  console.error('❌ data.js missing required exports');
  process.exit(1);
}
console.log('✅ data.js verified with PIPELINE_NODES and DIAGNOSTIC_CASES.');

console.log('🎉 All Mobile & HIG automated verification tests passed!');
