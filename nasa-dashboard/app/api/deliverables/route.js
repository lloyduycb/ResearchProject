import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const DELIVERABLES_DIR = path.resolve(process.cwd(), '..', 'assessment_deliverables');

export async function GET() {
    try {
        const files = await fs.readdir(DELIVERABLES_DIR);
        const mdFiles = files.filter(f => f.endsWith('.md'));
        return NextResponse.json(mdFiles);
    } catch (error) {
        return NextResponse.json({ error: 'Could not list deliverables' }, { status: 500 });
    }
}
