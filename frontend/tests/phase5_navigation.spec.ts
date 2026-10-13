import { test, expect } from '@playwright/test';

test.describe('Phase 5 - Dashboard & Explainability Multi-Page Navigation', () => {
  test('navigates seamlessly across Live Monitoring, Explainability, Trip Report, and Grad-CAM', async ({ page }) => {
    // 1. Visit Dashboard
    await page.goto('http://localhost:3000/');
    await expect(page.locator('.brand-title')).toHaveText('DrishtiDrive AI');

    // Screenshot 1: Default Live Monitoring
    await expect(page.locator('.card-title', { hasText: 'Live Driver Monitoring' })).toBeVisible();
    await page.screenshot({ path: '.agent/screenshots/page_live_monitoring.png', fullPage: true });

    // 2. Navigate to Explainability Tab
    const explainTab = page.locator('.nav-tab', { hasText: 'Explainability' });
    await expect(explainTab).toBeVisible();
    await explainTab.click();

    // Verify Explainability Content
    await expect(page.locator('.section-title', { hasText: 'Alert Explainability & Fusion Diagnostics' })).toBeVisible();
    await expect(page.locator('.gauge-name', { hasText: 'EAR (Eye Aspect)' })).toBeVisible();
    await expect(page.locator('.gauge-name', { hasText: 'PERCLOS (30s)' })).toBeVisible();
    await expect(page.locator('.gauge-name', { hasText: 'Head Nod (Pitch)' })).toBeVisible();
    await expect(page.locator('.gauge-name', { hasText: 'Yawn MAR' })).toBeVisible();
    await expect(page.locator('.stacked-bar')).toBeVisible();

    // Screenshot 2: Explainability View
    await page.screenshot({ path: '.agent/screenshots/page_explainability.png', fullPage: true });

    // 3. Navigate to Trip Report Tab
    const tripTab = page.locator('.nav-tab', { hasText: 'Trip Report' });
    await expect(tripTab).toBeVisible();
    await tripTab.click();

    // Verify Trip Report Content
    await expect(page.locator('.report-title', { hasText: 'Commercial Trip Safety Audit & Summary' })).toBeVisible();
    await expect(page.locator('.kpi-label', { hasText: 'Safety Risk Score' })).toBeVisible();
    await expect(page.locator('.kpi-label', { hasText: 'Driving Duration' })).toBeVisible();
    await expect(page.locator('.recs-title', { hasText: 'Offline Safety Recommendations' })).toBeVisible();

    // Screenshot 3: Trip Report View
    await page.screenshot({ path: '.agent/screenshots/page_trip_report.png', fullPage: true });

    // 4. Navigate to Grad-CAM Debug Tab
    const gradCamTab = page.locator('.nav-tab', { hasText: 'Grad-CAM Debug' });
    await expect(gradCamTab).toBeVisible();
    await gradCamTab.click();

    // Verify Grad-CAM Content
    await expect(page.locator('.gradcam-title', { hasText: 'Grad-CAM Model Attention Diagnostics' })).toBeVisible();
    await expect(page.locator('.card-badge', { hasText: 'Input Crop (64x64)' })).toBeVisible();
    await expect(page.locator('.card-badge', { hasText: 'Grad-CAM Saliency Overlay' })).toBeVisible();
    await expect(page.locator('.metrics-heading', { hasText: 'Inference Saliency Summary' })).toBeVisible();

    // Click sample toggle button
    const openSampleBtn = page.locator('.sample-btn', { hasText: 'Open Eye Sample' });
    await openSampleBtn.click();
    await expect(openSampleBtn).toHaveClass(/active/);

    // Screenshot 4: Grad-CAM Debug View
    await page.screenshot({ path: '.agent/screenshots/page_gradcam_debug.png', fullPage: true });
  });
});
