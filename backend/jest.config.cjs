module.exports = {
  testEnvironment: 'node',
  transform: {
    '^.+\\.jsx?$': 'babel-jest'
  },
  testMatch: ['**/tests/**/*.test.js'],
  moduleFileExtensions: ['js', 'json', 'jsx', 'node']
}
