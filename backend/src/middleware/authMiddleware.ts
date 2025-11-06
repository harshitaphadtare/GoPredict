import { Request, Response, NextFunction } from 'express';
import admin from '../config/firebaseAdmin'; // Import initialized admin SDK

// Extend the Express Request type to include our 'user' property
declare global {
    // eslint-disable-next-line @typescript-eslint/no-namespace
    namespace Express {
        interface Request {
            // The 'user' property will hold the decoded Firebase token
            user?: admin.auth.DecodedIdToken;
        }
    }
}

/**
 * Middleware to protect routes.
 * Verifies the Firebase ID token from the Authorization header.
 * If valid, attaches the decoded token (including user UID) to req.user.
 * If invalid, sends a 401 Unauthorized response.
 */
export const protect = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    let token: string | undefined;

    // 1. Check for Authorization header and ensure it starts with "Bearer "
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer ')) {
        try {
            // 2. Extract the token (the part *after* "Bearer ")
            token = req.headers.authorization.split(' ')[1];

            if (!token) {
                res.status(401).json({ message: 'Not authorized, token format is invalid' });
                return;
            }

            // 3. Verify the token using the Firebase Admin SDK
            const decodedToken = await admin.auth().verifyIdToken(token);

            // 4. Token is valid! Attach the decoded user info to the request object
            req.user = decodedToken;

            // 5. Pass control to the next function (the actual route handler)
            next();

        } catch (error) {
            // Token is invalid (expired, wrong signature, etc.)
            console.error('Token verification failed:', error);
            res.status(401).json({ message: 'Not authorized, token verification failed' });
        }
    } else {
        // No Authorization header found
        res.status(401).json({ message: 'Not authorized, no token provided' });
    }
};