import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const DELIVERABLES_DIR = path.resolve(process.cwd(), '..', 'assessment_deliverables');

export async function GET(request, { params }) {
    try {
        const { filename } = await params;
        const filePath = path.join(DELIVERABLES_DIR, filename);

        try {
            await fs.access(filePath);
        } catch {
            return NextResponse.json({ error: 'File not found' }, { status: 404 });
        }

        const content = await fs.readFile(filePath, 'utf8');
        return NextResponse.json({ content, filename });
    } catch (error) {
        return NextResponse.json({ error: 'Could not read deliverable' }, { status: 500 });
    }
}
