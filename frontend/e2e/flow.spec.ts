import { expect, test } from '@playwright/test'

test('answers to estimate, what-if and PDF', async ({ page }, info) => {
  const problems: string[] = []
  page.on('console', (message) => message.type() === 'error' && problems.push(message.text()))
  const shot = (name: string) =>
    page.screenshot({ path: `test-results/shots/${info.project.name}-${name}.png`, fullPage: true })
  const next = () => page.getByRole('button', { name: 'Next page' }).click()
  const choose = (group: RegExp, option: RegExp) =>
    page.getByRole('group', { name: group }).getByText(option).click()

  await page.goto('/')
  await shot('1-cover')
  await page.getByRole('button', { name: 'Open the notebook' }).click()

  await page.getByLabel('Age').fill('28')
  await choose(/Sex/, /^Male$/)
  await page.getByLabel('Height (cm)').fill('172')
  await page.getByLabel('Weight (kg)').fill('78')
  await next()

  await page.getByLabel('You fall asleep at').fill('01:00')
  await page.getByLabel('You wake up at').fill('06:00')
  await expect(page.getByText('5 hours of sleep')).toBeVisible()
  await shot('2-sleep')
  await next()

  await choose(/Smoking/, /I smoke/)
  await choose(/chew tobacco/, /^No$/)
  await choose(/Alcohol/, /15 or more/)
  await page.getByLabel('Moderate exercise').fill('30')
  await choose(/diet/, /^Fair$/)
  await shot('3-habits')
  await next()

  await page.getByRole('button', { name: 'Read my page' }).click()
  await expect(page.getByTestId('median-age')).toHaveText(/^\d{2}$/)
  await expect(page.getByText('No change yet')).toBeVisible()

  await choose(/^Smoking$/, /Quit/)
  await expect(page.getByText(/\+[\d.]+ years, middle estimate/)).toBeVisible()
  await page.waitForTimeout(2000)
  await shot('4-result')

  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download PDF' }).click()
  expect((await download).suggestedFilename()).toBe('life-estimate.pdf')

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(0)
  expect(problems).toEqual([])
})
