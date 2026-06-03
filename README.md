# Sage AI - Premium Enterprise Workspace Assistant

Sage AI is a high-performance, production-grade research and workspace assistant built on Cloudflare's global network. It leverages Durable Objects for stateful context and Cloudflare AI (Llama 3.1) for sophisticated enterprise-level reasoning.

## 🚀 Key Features

- **Stateful Intelligence**: Durable Object-backed conversation history for persistent, context-aware assistance.
- **Enterprise Speed**: Zero-latency global routing across Cloudflare's edge network.
- **Multi-Tool Integration**:
    - 🔍 **Global Search**: Real-time intelligence via DuckDuckGo API.
    - ⏰ **Smart Reminders**: Precision scheduling using Durable Object Alarms.
    - 🌐 **Env Awareness**: Browser-level context (timezone, locale) for personalized help.
- **Premium Aesthetics**: High-end UX with glassmorphic UI, fluid animations, and dark-mode elegance.

## 🏗 System Architecture

Sage AI is engineered for scalability and low-latency:
- **State Layer**: Durable Objects provide 100% consistency for user sessions.
- **Inference Layer**: Direct integration with `@cf/meta/llama-3.1-8b-instruct`.
- **UI Layer**: React-based frontend with server-sent event (SSE) streaming.

For more details, see [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md).

## 🛠 Getting Started

### Prerequisites
- Node.js & npm
- Cloudflare Account with Workers & AI enabled.

### Installation
```bash
# Install dependencies
npm install

# Local development
npm run dev
```

### Deployment
```bash
# Build and deploy to Cloudflare
npm run deploy
```

## 📚 Documentation

- [System Architecture](./SYSTEM_ARCHITECTURE.md)
- [API Reference](./API_REFERENCE.md)
- [Security & Compliance](./README.md)

## ⚖️ License
MIT. Built with ❤️ on Cloudflare.
