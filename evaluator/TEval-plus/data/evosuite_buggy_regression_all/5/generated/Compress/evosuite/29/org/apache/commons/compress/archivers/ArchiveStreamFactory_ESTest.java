/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:50:43 GMT 2023
 */

package org.apache.commons.compress.archivers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.ArchiveOutputStream;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.ar.ArArchiveOutputStream;
import org.apache.commons.compress.archivers.jar.JarArchiveOutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArchiveStreamFactory_ESTest extends ArchiveStreamFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      String string0 = archiveStreamFactory0.getEntryEncoding();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      archiveStreamFactory0.setEntryEncoding("ar");
      assertEquals("ar", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      // Undeclared exception!
      try { 
        archiveStreamFactory0.setEntryEncoding((String) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Cannot overide encoding set by the constructor
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        archiveStreamFactory0.createArchiveInputStream("arj", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.arj.ArjArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 1, (-1611));
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((String) null, (InputStream) byteArrayInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Archivername must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("ar", (InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // InputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream((byte)126);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("ar", (InputStream) pipedInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 22);
      try { 
        archiveStreamFactory0.createArchiveInputStream("f,(qae/%ei2~y|+=swg", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: f,(qae/%ei2~y|+=swg not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      assertNotNull(archiveStreamFactory0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-1443), (-1443));
      assertNotNull(byteArrayInputStream0);
      assertEquals(6, byteArray0.length);
      assertEquals((-1443), byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream("arj", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.arj.ArjArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(9, byteArray0.length);
      assertEquals(9, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0, 1744);
      assertNotNull(bufferedInputStream0);
      assertEquals(9, byteArray0.length);
      assertEquals(9, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) bufferedInputStream0);
      assertNotNull(archiveInputStream0);
      assertEquals(9, byteArray0.length);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(9, byteArrayInputStream0.available());
      assertEquals(0L, archiveInputStream0.getBytesRead());
      assertEquals(0, archiveInputStream0.getCount());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(9, byteArray0.length);
      assertEquals(9, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("ar");
      assertNotNull(archiveStreamFactory0);
      assertEquals("ar", archiveStreamFactory0.getEntryEncoding());
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0, 1744);
      assertNotNull(bufferedInputStream0);
      assertEquals(9, byteArray0.length);
      assertEquals(9, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) bufferedInputStream0);
      assertNotNull(archiveInputStream0);
      assertEquals(9, byteArray0.length);
      assertEquals(9, byteArrayInputStream0.available());
      assertEquals("ar", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0L, archiveInputStream0.getBytesRead());
      assertEquals(0, archiveInputStream0.getCount());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      assertNotNull(archiveStreamFactory0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      assertNotNull(pipedOutputStream0);
      
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 22);
      assertNotNull(pipedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      
      TarArchiveInputStream tarArchiveInputStream0 = (TarArchiveInputStream)archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) pipedInputStream0);
      assertNotNull(tarArchiveInputStream0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, pipedInputStream0.available());
      assertEquals(0, tarArchiveInputStream0.getCount());
      assertFalse(tarArchiveInputStream0.markSupported());
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0, tarArchiveInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      assertNotNull(pipedOutputStream0);
      
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 22);
      assertNotNull(pipedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      
      TarArchiveInputStream tarArchiveInputStream0 = (TarArchiveInputStream)archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) pipedInputStream0);
      assertNotNull(tarArchiveInputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, pipedInputStream0.available());
      assertEquals(0, tarArchiveInputStream0.getCount());
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0, tarArchiveInputStream0.available());
      assertFalse(tarArchiveInputStream0.markSupported());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(7, byteArray0.length);
      assertEquals(7, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) byteArrayInputStream0);
      assertNotNull(archiveInputStream0);
      assertEquals(7, byteArray0.length);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(7, byteArrayInputStream0.available());
      assertEquals(0L, archiveInputStream0.getBytesRead());
      assertEquals(0, archiveInputStream0.getCount());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      assertNotNull(archiveStreamFactory0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      assertNotNull(pipedOutputStream0);
      
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 22);
      assertNotNull(pipedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) pipedInputStream0);
      assertNotNull(archiveInputStream0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, pipedInputStream0.available());
      assertEquals(0L, archiveInputStream0.getBytesRead());
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      assertNotNull(archiveStreamFactory0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(6, byteArray0.length);
      assertEquals(6, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0, 32);
      assertNotNull(bufferedInputStream0);
      assertEquals(6, byteArray0.length);
      assertEquals(6, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) bufferedInputStream0);
      assertNotNull(archiveInputStream0);
      assertEquals(6, byteArray0.length);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      assertEquals(6, byteArrayInputStream0.available());
      assertEquals(0L, archiveInputStream0.getBytesRead());
      assertEquals(0, archiveInputStream0.getCount());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(7, byteArray0.length);
      assertEquals(7, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) byteArrayInputStream0);
      assertNotNull(archiveInputStream0);
      assertEquals(7, byteArray0.length);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(7, byteArrayInputStream0.available());
      assertEquals(0, archiveInputStream0.getCount());
      assertEquals(0L, archiveInputStream0.getBytesRead());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      File file0 = MockFile.createTempFile("ss6oCJ", "cpio");
      assertNotNull(file0);
      assertFalse(file0.isHidden());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertTrue(file0.canWrite());
      assertEquals(0L, file0.getUsableSpace());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canExecute());
      assertFalse(file0.isDirectory());
      assertEquals(0L, file0.length());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals("ss6oCJ0cpio", file0.getName());
      assertTrue(file0.exists());
      assertEquals(1392409281320L, file0.lastModified());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals("/tmp/ss6oCJ0cpio", file0.toString());
      
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      assertNotNull(mockFileInputStream0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream("dump", (InputStream) mockFileInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // unexpected EOF
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("tar");
      assertNotNull(archiveStreamFactory0);
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 2993, (byte) (-8));
      assertNotNull(byteArrayInputStream0);
      assertEquals(7, byteArray0.length);
      assertEquals((-2986), byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream("dump", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // unexpected EOF
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(7, byteArray0.length);
      assertEquals(7, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream("7z", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // The 7z doesn't support streaming.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      
      TarArchiveOutputStream tarArchiveOutputStream0 = (TarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("tar", byteArrayOutputStream0);
      assertEquals(3, TarArchiveOutputStream.LONGFILE_POSIX);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(2, TarArchiveOutputStream.BIGNUMBER_POSIX);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
      assertNotNull(tarArchiveOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
      assertEquals(0, tarArchiveOutputStream0.getCount());
      assertEquals(0L, tarArchiveOutputStream0.getBytesWritten());
      
      ZipArchiveOutputStream zipArchiveOutputStream0 = (ZipArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("zip", tarArchiveOutputStream0);
      assertEquals(3, TarArchiveOutputStream.LONGFILE_POSIX);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(2, TarArchiveOutputStream.BIGNUMBER_POSIX);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
      assertEquals(8, ZipArchiveOutputStream.DEFLATED);
      assertEquals(2048, ZipArchiveOutputStream.EFS_FLAG);
      assertEquals(0, ZipArchiveOutputStream.STORED);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
      assertNotNull(zipArchiveOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
      assertEquals(0, tarArchiveOutputStream0.getCount());
      assertEquals(0L, tarArchiveOutputStream0.getBytesWritten());
      assertEquals(0L, zipArchiveOutputStream0.getBytesWritten());
      assertEquals(0, zipArchiveOutputStream0.getCount());
      assertFalse(zipArchiveOutputStream0.isSeekable());
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream((String) null, byteArrayOutputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Archivername must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("n9tPX~", (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // OutputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      assertNotNull(archiveStreamFactory0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      
      ArArchiveOutputStream arArchiveOutputStream0 = (ArArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("ar", byteArrayOutputStream0);
      assertEquals(0, ArArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(1, ArArchiveOutputStream.LONGFILE_BSD);
      assertNotNull(arArchiveOutputStream0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0L, arArchiveOutputStream0.getBytesWritten());
      assertEquals(0, arArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      
      TarArchiveOutputStream tarArchiveOutputStream0 = (TarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("tar", byteArrayOutputStream0);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
      assertEquals(2, TarArchiveOutputStream.BIGNUMBER_POSIX);
      assertEquals(3, TarArchiveOutputStream.LONGFILE_POSIX);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
      assertNotNull(tarArchiveOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0L, tarArchiveOutputStream0.getBytesWritten());
      assertEquals(0, tarArchiveOutputStream0.getCount());
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
      
      ArchiveStreamFactory archiveStreamFactory1 = new ArchiveStreamFactory("ar");
      assertFalse(archiveStreamFactory1.equals((Object)archiveStreamFactory0));
      assertNotNull(archiveStreamFactory1);
      assertEquals("ar", archiveStreamFactory1.getEntryEncoding());
      
      ZipArchiveOutputStream zipArchiveOutputStream0 = (ZipArchiveOutputStream)archiveStreamFactory1.createArchiveOutputStream("zip", tarArchiveOutputStream0);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
      assertEquals(2, TarArchiveOutputStream.BIGNUMBER_POSIX);
      assertEquals(3, TarArchiveOutputStream.LONGFILE_POSIX);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
      assertEquals(0, ZipArchiveOutputStream.STORED);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
      assertEquals(8, ZipArchiveOutputStream.DEFLATED);
      assertEquals(2048, ZipArchiveOutputStream.EFS_FLAG);
      assertFalse(archiveStreamFactory0.equals((Object)archiveStreamFactory1));
      assertFalse(archiveStreamFactory1.equals((Object)archiveStreamFactory0));
      assertNotSame(archiveStreamFactory0, archiveStreamFactory1);
      assertNotSame(archiveStreamFactory1, archiveStreamFactory0);
      assertNotNull(zipArchiveOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0L, tarArchiveOutputStream0.getBytesWritten());
      assertEquals(0, tarArchiveOutputStream0.getCount());
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
      assertEquals("ar", archiveStreamFactory1.getEntryEncoding());
      assertFalse(zipArchiveOutputStream0.isSeekable());
      assertEquals(0L, zipArchiveOutputStream0.getBytesWritten());
      assertEquals(0, zipArchiveOutputStream0.getCount());
      assertEquals("ar", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      
      JarArchiveOutputStream jarArchiveOutputStream0 = (JarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("jar", byteArrayOutputStream0);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
      assertEquals(0, ZipArchiveOutputStream.STORED);
      assertEquals(8, ZipArchiveOutputStream.DEFLATED);
      assertEquals(2048, ZipArchiveOutputStream.EFS_FLAG);
      assertNotNull(jarArchiveOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals(0L, jarArchiveOutputStream0.getBytesWritten());
      assertEquals(0, jarArchiveOutputStream0.getCount());
      assertEquals("UTF8", jarArchiveOutputStream0.getEncoding());
      assertFalse(jarArchiveOutputStream0.isSeekable());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("jip");
      assertNotNull(mockFileOutputStream0);
      
      ObjectOutputStream objectOutputStream0 = new ObjectOutputStream(mockFileOutputStream0);
      assertNotNull(objectOutputStream0);
      
      try { 
        archiveStreamFactory0.createArchiveOutputStream("7z", objectOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // The 7z doesn't support streaming.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      assertNotNull(archiveStreamFactory0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("cpio", byteArrayOutputStream0);
      assertNotNull(archiveOutputStream0);
      assertEquals("arj", archiveStreamFactory0.getEntryEncoding());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals(0, archiveOutputStream0.getCount());
      assertEquals(0L, archiveOutputStream0.getBytesWritten());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("cpio", byteArrayOutputStream0);
      assertNotNull(archiveOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals(0L, archiveOutputStream0.getBytesWritten());
      assertEquals(0, archiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      assertNotNull(byteArrayOutputStream0);
      assertEquals("", byteArrayOutputStream0.toString());
      assertEquals(0, byteArrayOutputStream0.size());
      
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      try { 
        archiveStreamFactory0.createArchiveOutputStream("qljWh,t", byteArrayOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: qljWh,t not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNotNull(archiveStreamFactory0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(9, byteArray0.length);
      assertEquals(9, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream((InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // No Archiver found for the stream signature
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("g");
      assertNotNull(archiveStreamFactory0);
      assertEquals("g", archiveStreamFactory0.getEntryEncoding());
      
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Stream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("");
      assertNotNull(archiveStreamFactory0);
      assertEquals("", archiveStreamFactory0.getEntryEncoding());
      
      PipedInputStream pipedInputStream0 = new PipedInputStream(2460);
      assertNotNull(pipedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((InputStream) pipedInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Mark is not supported.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }
}