export default {
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: [
    '**/__tests__/**/*.js',
    '**/*.(test|spec).js'
  ],
  collectCoverageFrom: [
    'routes/**/*.js',
    'server.js',
    '!**/node_modules/**',
    '!**/coverage/**'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  transform: {},
  testTimeout: 10000
}
