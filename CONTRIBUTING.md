# Contributing to GoPredict

Thanks for your interest in contributing! This guide will help you get started.

## Ground Rules
- **Be kind and respectful.** Follow our Code of Conduct.
- **Open an issue first** for major changes to discuss the approach.
- **Add tests** for any bug fix or new feature.
- **Keep PRs focused** and small when possible.

## Project Setup

### Frontend
```bash
cd frontend
npm install
npm run dev
npm run test:run
```

### Backend
```bash
cd backend
npm install
npm start
npm test
```

### Python ML (optional)
```bash
pip install -r requirements.txt
# Add tests under tests/ if contributing Python code
```

## Branching & Commits
- Create a feature branch from `main` or `develop`:
  - `feature/<short-description>`
  - `fix/<short-description>`
- Use conventional-style commit messages when possible (e.g., `feat: add vitest setup`).

## Testing
- Frontend: Vitest + React Testing Library (`frontend/src/test/*`)
- Backend: Jest + Supertest (`backend/tests/*`)
- Ensure `npm run test:coverage` passes in both `frontend/` and `backend/`.

## Linting & Formatting
- Use Prettier defaults (Vite + React).
- Keep code idiomatic and typed on the frontend.

## Pull Requests
- Fill in the PR template.
- Link related issues.
- Describe the change, screenshots if UI.
- Checklist must pass: tests, CI, and review comments.

## Releases
- Maintainers use GitHub Releases and tags.

## Questions?
Open a Discussion or an Issue on GitHub.
