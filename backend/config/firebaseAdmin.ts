import * as admin from 'firebase-admin';
import dotenv from 'dotenv';
import path from 'path'; // Import path module

dotenv.config(); // Load environment variables

const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_KEY_PATH;

if (!serviceAccountPath) {
    console.error(' FATAL ERROR: FIREBASE_SERVICE_ACCOUNT_KEY_PATH is not defined in .env file');
    process.exit(1);
}

// Construct the absolute path from the project's root directory (where package.json is)
const absolutePath = path.resolve(process.cwd(), serviceAccountPath);

try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const serviceAccount = require(absolutePath); // Use the absolute path

    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount)
    });

    console.log(' Firebase Admin SDK Initialized Successfully');

} catch (error: any) {
    console.error('Firebase Admin SDK Initialization Error:', error.message);
    console.error(`Attempted to load key from: ${absolutePath}`);
    console.error('Ensure the path in .env is correct (relative to project root) and the JSON file exists.');
    process.exit(1);
}

export default admin;