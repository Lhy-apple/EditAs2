/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:23:41 GMT 2023
 */

package org.apache.commons.compress.archivers.cpio;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CpioArchiveInputStream_ESTest extends CpioArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)6, (byte)6);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte)6);
      int int0 = cpioArchiveInputStream0.available();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      File file0 = MockFile.createTempFile("070701", "070701");
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(mockFileInputStream0, 199, "070701");
      try { 
        cpioArchiveInputStream0.getNextEntry();
        fail("Expecting exception: EOFException");
      
      } catch(EOFException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 2, (byte)0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, 2);
      long long0 = cpioArchiveInputStream0.skip(656L);
      assertEquals(0L, long0);
      
      int int0 = cpioArchiveInputStream0.available();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile("070701", "070701");
      assertEquals("/tmp/0707010070701", file0.toString());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.exists());
      assertTrue(file0.canExecute());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canWrite());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertFalse(file0.isDirectory());
      assertEquals("0707010070701", file0.getName());
      assertNotNull(file0);
      
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      assertNotNull(mockFileInputStream0);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(mockFileInputStream0, 199, "070701");
      assertEquals("/tmp/0707010070701", file0.toString());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.exists());
      assertTrue(file0.canExecute());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canWrite());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertFalse(file0.isDirectory());
      assertEquals("0707010070701", file0.getName());
      assertEquals(0, mockFileInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertNotNull(cpioArchiveInputStream0);
      
      cpioArchiveInputStream0.close();
      assertEquals("/tmp/0707010070701", file0.toString());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.exists());
      assertTrue(file0.canExecute());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canWrite());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertFalse(file0.isDirectory());
      assertEquals("0707010070701", file0.getName());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      
      cpioArchiveInputStream0.close();
      assertEquals("/tmp/0707010070701", file0.toString());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.exists());
      assertTrue(file0.canExecute());
      assertTrue(file0.isAbsolute());
      assertTrue(file0.canWrite());
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertFalse(file0.isDirectory());
      assertEquals("0707010070701", file0.getName());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = MockFile.createTempFile("070701", "070701");
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.exists());
      assertEquals(1392409281320L, file0.lastModified());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals("/tmp/0707010070701", file0.toString());
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertTrue(file0.canExecute());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isAbsolute());
      assertEquals("0707010070701", file0.getName());
      assertFalse(file0.isDirectory());
      assertNotNull(file0);
      
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      assertNotNull(mockFileInputStream0);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(mockFileInputStream0, (byte) (-126), (String) null);
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.exists());
      assertEquals(1392409281320L, file0.lastModified());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals("/tmp/0707010070701", file0.toString());
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertTrue(file0.canExecute());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isAbsolute());
      assertEquals("0707010070701", file0.getName());
      assertFalse(file0.isDirectory());
      assertEquals(0, mockFileInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertNotNull(cpioArchiveInputStream0);
      
      cpioArchiveInputStream0.close();
      assertEquals(0L, file0.getFreeSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.exists());
      assertEquals(1392409281320L, file0.lastModified());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals("/tmp/0707010070701", file0.toString());
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.canRead());
      assertTrue(file0.canExecute());
      assertEquals(0L, file0.getUsableSpace());
      assertFalse(file0.isHidden());
      assertTrue(file0.isAbsolute());
      assertEquals("0707010070701", file0.getName());
      assertFalse(file0.isDirectory());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      
      try { 
        cpioArchiveInputStream0.skip(29127L);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertEquals(11, byteArrayInputStream0.available());
      assertNotNull(byteArrayInputStream0);
      assertEquals(11, byteArray0.length);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0);
      assertEquals(11, byteArrayInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertNotNull(cpioArchiveInputStream0);
      assertEquals(11, byteArray0.length);
      
      try { 
        cpioArchiveInputStream0.getNextCPIOEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unknown magic [\u0000\u0000\u0000\u0000\u0000\u0000]. Occured at byte: 6
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream((InputStream) null, "070701");
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertNotNull(cpioArchiveInputStream0);
      
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, (-1789569698), (int) (byte)6);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("vv\uFFFD\uFFFD\uFFFD\uFFFD", "*R/uv Q':#");
      assertNotNull(mockFile0);
      
      File file0 = MockFile.createTempFile("*R/uv Q':#", "*R/uv Q':#", (File) mockFile0);
      assertTrue(mockFile0.canExecute());
      assertTrue(mockFile0.isAbsolute());
      assertTrue(mockFile0.canRead());
      assertEquals("uv Q':#", mockFile0.getName());
      assertFalse(mockFile0.isFile());
      assertTrue(mockFile0.isDirectory());
      assertEquals(0L, mockFile0.getUsableSpace());
      assertEquals(0L, mockFile0.getFreeSpace());
      assertTrue(mockFile0.canWrite());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R", mockFile0.getParent());
      assertFalse(mockFile0.isHidden());
      assertEquals(0L, mockFile0.length());
      assertEquals(0L, mockFile0.getTotalSpace());
      assertEquals(1392409281320L, mockFile0.lastModified());
      assertTrue(mockFile0.exists());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R/uv Q':#", mockFile0.toString());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.canExecute());
      assertFalse(file0.isDirectory());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R/uv Q':#/*R/uv Q':#0*R", file0.getParent());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getFreeSpace());
      assertTrue(file0.canWrite());
      assertTrue(file0.isAbsolute());
      assertEquals(0L, file0.getUsableSpace());
      assertTrue(file0.canRead());
      assertTrue(file0.isFile());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R/uv Q':#/*R/uv Q':#0*R/uv Q':#", file0.toString());
      assertEquals("uv Q':#", file0.getName());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.exists());
      assertNotSame(mockFile0, file0);
      assertNotSame(file0, mockFile0);
      assertNotNull(file0);
      assertFalse(file0.equals((Object)mockFile0));
      
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      assertNotNull(mockFileInputStream0);
      assertFalse(mockFile0.equals((Object)file0));
      assertFalse(file0.equals((Object)mockFile0));
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(mockFileInputStream0, (-278793704));
      assertTrue(mockFile0.canExecute());
      assertTrue(mockFile0.isAbsolute());
      assertTrue(mockFile0.canRead());
      assertEquals("uv Q':#", mockFile0.getName());
      assertFalse(mockFile0.isFile());
      assertTrue(mockFile0.isDirectory());
      assertEquals(0L, mockFile0.getUsableSpace());
      assertEquals(0L, mockFile0.getFreeSpace());
      assertTrue(mockFile0.canWrite());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R", mockFile0.getParent());
      assertFalse(mockFile0.isHidden());
      assertEquals(0L, mockFile0.length());
      assertEquals(0L, mockFile0.getTotalSpace());
      assertEquals(1392409281320L, mockFile0.lastModified());
      assertTrue(mockFile0.exists());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R/uv Q':#", mockFile0.toString());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.canExecute());
      assertFalse(file0.isDirectory());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R/uv Q':#/*R/uv Q':#0*R", file0.getParent());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getFreeSpace());
      assertTrue(file0.canWrite());
      assertTrue(file0.isAbsolute());
      assertEquals(0L, file0.getUsableSpace());
      assertTrue(file0.canRead());
      assertTrue(file0.isFile());
      assertEquals("/data/swf/zenodo_replication_package_new/vv\uFFFD\uFFFD\uFFFD\uFFFD/*R/uv Q':#/*R/uv Q':#0*R/uv Q':#", file0.toString());
      assertEquals("uv Q':#", file0.getName());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.exists());
      assertEquals(0, mockFileInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertNotNull(cpioArchiveInputStream0);
      assertFalse(mockFile0.equals((Object)file0));
      assertFalse(file0.equals((Object)mockFile0));
      
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read((byte[]) null, 1091, (-278793704));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      assertNotNull(pipedOutputStream0);
      
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      assertEquals(0, pipedInputStream0.available());
      assertNotNull(pipedInputStream0);
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(pipedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      assertNotNull(bufferedInputStream0);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(bufferedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertNotNull(cpioArchiveInputStream0);
      
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, (int) (byte)54, 374);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertEquals(4, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(4, byteArray0.length);
      
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0, 512);
      assertEquals(4, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertNotNull(pushbackInputStream0);
      assertEquals(4, byteArray0.length);
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(pushbackInputStream0);
      assertEquals(4, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertNotNull(bufferedInputStream0);
      assertEquals(4, byteArray0.length);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(bufferedInputStream0, 3, "org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream");
      assertEquals(4, byteArrayInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertNotNull(cpioArchiveInputStream0);
      assertEquals(4, byteArray0.length);
      
      int int0 = cpioArchiveInputStream0.read(byteArray0, (int) (byte)0, (int) (byte)0);
      assertEquals(4, byteArrayInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertEquals(0, int0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(4, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 381, (byte)0);
      assertEquals((-373), byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertNotNull(byteArrayInputStream0);
      assertEquals(8, byteArray0.length);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte)0);
      assertEquals((-373), byteArrayInputStream0.available());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertNotNull(cpioArchiveInputStream0);
      assertEquals(8, byteArray0.length);
      
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.skip((byte) (-5));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // negative skip length
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      assertEquals(0, pipedInputStream0.available());
      assertNotNull(pipedInputStream0);
      
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(pipedInputStream0, pipedInputStream0);
      assertEquals(0, pipedInputStream0.available());
      assertNotNull(sequenceInputStream0);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(sequenceInputStream0, (byte)0);
      assertEquals(0, pipedInputStream0.available());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertNotNull(cpioArchiveInputStream0);
      
      long long0 = cpioArchiveInputStream0.skip((byte)0);
      assertEquals(0, pipedInputStream0.available());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      File file0 = MockFile.createTempFile("INODE", (String) null);
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertTrue(file0.canRead());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.isAbsolute());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getUsableSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.exists());
      assertEquals("INODE0.tmp", file0.getName());
      assertEquals(0L, file0.getFreeSpace());
      assertFalse(file0.isDirectory());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.canExecute());
      assertEquals("/tmp/INODE0.tmp", file0.toString());
      assertNotNull(file0);
      
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      assertNotNull(mockFileInputStream0);
      
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(mockFileInputStream0, 6, "INODE");
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertTrue(file0.canRead());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.isAbsolute());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getUsableSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.exists());
      assertEquals("INODE0.tmp", file0.getName());
      assertEquals(0L, file0.getFreeSpace());
      assertFalse(file0.isDirectory());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.canExecute());
      assertEquals("/tmp/INODE0.tmp", file0.toString());
      assertEquals(0, mockFileInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertNotNull(cpioArchiveInputStream0);
      
      long long0 = cpioArchiveInputStream0.skip(13107L);
      assertTrue(file0.canWrite());
      assertTrue(file0.isFile());
      assertTrue(file0.canRead());
      assertEquals("/tmp", file0.getParent());
      assertTrue(file0.isAbsolute());
      assertFalse(file0.isHidden());
      assertEquals(0L, file0.getUsableSpace());
      assertEquals(0L, file0.length());
      assertTrue(file0.exists());
      assertEquals("INODE0.tmp", file0.getName());
      assertEquals(0L, file0.getFreeSpace());
      assertFalse(file0.isDirectory());
      assertEquals(0L, file0.getTotalSpace());
      assertEquals(1392409281320L, file0.lastModified());
      assertTrue(file0.canExecute());
      assertEquals("/tmp/INODE0.tmp", file0.toString());
      assertEquals(0, mockFileInputStream0.available());
      assertEquals(0, cpioArchiveInputStream0.getCount());
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      boolean boolean0 = CpioArchiveInputStream.matches(byteArray0, 186);
      assertFalse(boolean0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(3, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      boolean boolean0 = CpioArchiveInputStream.matches(byteArray0, 0);
      assertFalse(boolean0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(7, byteArray0.length);
  }
}
