##  Summary
##  Problem & Motivation
Fixes: #


##  What Changed
### Frontend
- **`src/components/FileName.tsx`**:
    - Implemented...
    - Changed...

### Backend
- **`src/routes/api/endpoint.js`**:
    - Added a new proxy for...
    - Modified the response to...


##  Screenshots
### Before
### After
##  How to Test
### Local Setup
1. **Backend:**
   ```bash
   cd backend
   npm install
   ```
   Create a `.env` file in the `backend` directory with the following variables:
   ```env
   # Description of what this key is for
   API_KEY_NAME=your_key_here
   ```

2. **Frontend:**
   ```bash
   cd frontend
   pnpm install
   ```
   Create a `.env` file in the `frontend` directory if needed:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

### Manual Testing Steps
1. Navigate to the page/feature (e.g., `/dashboard`).
2. Perform a specific action (e.g., "Run a prediction from A to B").
3. **Expected Behavior:** Describe what should happen (e.g., "The map should update with a road-following polyline.").
4. Check the browser's **Network Tab** for API calls (e.g., "Look for a `POST` to `/api/routing` with a `200` status").
5. Check the **Backend Console** for logs.


##  Notes for Reviewers
##  Checklist
- [ ] My code follows the project's coding standards.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have made corresponding changes to the documentation.
- [ ] My changes generate no new warnings.
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] I have verified that no API keys or other secrets are committed.
- [ ] I have updated `.env.example` with any new environment variables.