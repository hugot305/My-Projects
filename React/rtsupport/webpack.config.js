module.exports = {
    entry: './index.js',
    output: {
      path: __dirname,
      filename: 'bundle.js'
    },
    module: {
      rules: [
        {
          test: /\.jsx?$/,
          loader: 'babel-loader',
          exclude: /node_modules/,
          query: {
            presets: ['@babel/env', '@babel/react']
          }
        }
      ]
    }
  }