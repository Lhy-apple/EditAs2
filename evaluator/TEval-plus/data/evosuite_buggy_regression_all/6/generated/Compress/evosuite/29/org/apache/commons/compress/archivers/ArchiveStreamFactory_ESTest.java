/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:12:17 GMT 2023
 */

package org.apache.commons.compress.archivers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.SequenceInputStream;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Enumeration;
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
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArchiveStreamFactory_ESTest extends ArchiveStreamFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("0f5_Qib@yF");
      String string0 = archiveStreamFactory0.getEntryEncoding();
      assertEquals("0f5_Qib@yF", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      archiveStreamFactory0.setEntryEncoding("TD7s=%+kTKMS#^");
      MockFile mockFile0 = new MockFile("cpio", "xz-.!B");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("zip", mockFileOutputStream0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // TD7s=%+kTKMS#^
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("");
      // Undeclared exception!
      try { 
        archiveStreamFactory0.setEntryEncoding("");
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
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) pipedInputStream0);
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((InputStream) archiveInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Mark is not supported.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("OutputStream must not be null.", (InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // InputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      assertNotNull(sequenceInputStream0);
      
      TarArchiveInputStream tarArchiveInputStream0 = (TarArchiveInputStream)archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) sequenceInputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0, tarArchiveInputStream0.available());
      assertFalse(tarArchiveInputStream0.markSupported());
      assertNotNull(tarArchiveInputStream0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("tar");
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      assertNotNull(sequenceInputStream0);
      
      TarArchiveInputStream tarArchiveInputStream0 = (TarArchiveInputStream)archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) sequenceInputStream0);
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, tarArchiveInputStream0.available());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(tarArchiveInputStream0.markSupported());
      assertNotNull(tarArchiveInputStream0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null, 684);
      assertNotNull(bufferedInputStream0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream("SlgT\"E-r", (InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: SlgT\"E-r not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream("7z");
      assertNotNull(mockPrintStream0);
      
      TarArchiveOutputStream tarArchiveOutputStream0 = (TarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("tar", mockPrintStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0, tarArchiveOutputStream0.getCount());
      assertEquals(0L, tarArchiveOutputStream0.getBytesWritten());
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
      assertNotNull(tarArchiveOutputStream0);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
      assertEquals(2, TarArchiveOutputStream.BIGNUMBER_POSIX);
      assertEquals(3, TarArchiveOutputStream.LONGFILE_POSIX);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(9578);
      assertEquals(0, byteArrayOutputStream0.size());
      assertEquals("", byteArrayOutputStream0.toString());
      assertNotNull(byteArrayOutputStream0);
      
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
  public void test10()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory((String) null);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("Q", (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // OutputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream");
      assertEquals("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockFile mockFile0 = new MockFile("arj", "cpio");
      assertNotNull(mockFile0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      assertNotNull(mockPrintStream0);
      
      ArArchiveOutputStream arArchiveOutputStream0 = (ArArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("ar", mockPrintStream0);
      assertEquals("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream", archiveStreamFactory0.getEntryEncoding());
      assertEquals("cpio", mockFile0.getName());
      assertTrue(mockFile0.canExecute());
      assertTrue(mockFile0.isAbsolute());
      assertTrue(mockFile0.canRead());
      assertTrue(mockFile0.isFile());
      assertEquals(0L, mockFile0.length());
      assertEquals(1392409281320L, mockFile0.lastModified());
      assertTrue(mockFile0.exists());
      assertEquals(0L, mockFile0.getTotalSpace());
      assertEquals("/data/lhy/TEval-plus/arj/cpio", mockFile0.toString());
      assertFalse(mockFile0.isDirectory());
      assertEquals(0L, mockFile0.getUsableSpace());
      assertEquals(0L, mockFile0.getFreeSpace());
      assertTrue(mockFile0.canWrite());
      assertFalse(mockFile0.isHidden());
      assertEquals("/data/lhy/TEval-plus/arj", mockFile0.getParent());
      assertEquals(0L, arArchiveOutputStream0.getBytesWritten());
      assertEquals(0, arArchiveOutputStream0.getCount());
      assertNotNull(arArchiveOutputStream0);
      assertEquals(0, ArArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(1, ArArchiveOutputStream.LONGFILE_BSD);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockFile mockFile0 = new MockFile("cpio", "xz-.!B");
      assertNotNull(mockFile0);
      
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      assertNotNull(mockFileOutputStream0);
      
      ZipArchiveOutputStream zipArchiveOutputStream0 = (ZipArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("zip", mockFileOutputStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0L, mockFile0.getUsableSpace());
      assertFalse(mockFile0.isHidden());
      assertEquals(0L, mockFile0.length());
      assertFalse(mockFile0.isDirectory());
      assertTrue(mockFile0.canWrite());
      assertEquals("/data/lhy/TEval-plus/cpio/xz-.!B", mockFile0.toString());
      assertEquals(0L, mockFile0.getFreeSpace());
      assertTrue(mockFile0.isAbsolute());
      assertTrue(mockFile0.canExecute());
      assertEquals(0L, mockFile0.getTotalSpace());
      assertEquals("/data/lhy/TEval-plus/cpio", mockFile0.getParent());
      assertTrue(mockFile0.exists());
      assertEquals(1392409281320L, mockFile0.lastModified());
      assertEquals("xz-.!B", mockFile0.getName());
      assertTrue(mockFile0.isFile());
      assertTrue(mockFile0.canRead());
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
      assertFalse(zipArchiveOutputStream0.isSeekable());
      assertEquals(0, zipArchiveOutputStream0.getCount());
      assertEquals(0L, zipArchiveOutputStream0.getBytesWritten());
      assertNotNull(zipArchiveOutputStream0);
      assertEquals(2048, ZipArchiveOutputStream.EFS_FLAG);
      assertEquals(8, ZipArchiveOutputStream.DEFLATED);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
      assertEquals(0, ZipArchiveOutputStream.STORED);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream("7z");
      assertNotNull(mockPrintStream0);
      
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("cpio", mockPrintStream0);
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertEquals(0L, archiveOutputStream0.getBytesWritten());
      assertEquals(0, archiveOutputStream0.getCount());
      assertNotNull(archiveOutputStream0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("tar");
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream("7z");
      assertNotNull(mockPrintStream0);
      
      TarArchiveOutputStream tarArchiveOutputStream0 = (TarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("tar", mockPrintStream0);
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0L, tarArchiveOutputStream0.getBytesWritten());
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
      assertEquals(0, tarArchiveOutputStream0.getCount());
      assertNotNull(tarArchiveOutputStream0);
      assertEquals(2, TarArchiveOutputStream.BIGNUMBER_POSIX);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
      assertEquals(3, TarArchiveOutputStream.LONGFILE_POSIX);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
      assertEquals(1, TarArchiveOutputStream.BIGNUMBER_STAR);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("always");
      assertEquals("always", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream("always");
      assertNotNull(mockPrintStream0);
      
      JarArchiveOutputStream jarArchiveOutputStream0 = (JarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("jar", mockPrintStream0);
      assertEquals("always", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0L, jarArchiveOutputStream0.getBytesWritten());
      assertEquals(0, jarArchiveOutputStream0.getCount());
      assertEquals("UTF8", jarArchiveOutputStream0.getEncoding());
      assertFalse(jarArchiveOutputStream0.isSeekable());
      assertNotNull(jarArchiveOutputStream0);
      assertEquals(2048, ZipArchiveOutputStream.EFS_FLAG);
      assertEquals(8, ZipArchiveOutputStream.DEFLATED);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
      assertEquals(0, ZipArchiveOutputStream.STORED);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream("dump");
      assertNotNull(mockPrintStream0);
      
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockPrintStream0, 60);
      assertNotNull(bufferedOutputStream0);
      
      try { 
        archiveStreamFactory0.createArchiveOutputStream("Cannot overide encoding set by the constructor", bufferedOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: Cannot overide encoding set by the constructor not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("tar");
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      MockPrintStream mockPrintStream0 = new MockPrintStream("7z");
      assertNotNull(mockPrintStream0);
      
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("cpio", mockPrintStream0);
      assertEquals("tar", archiveStreamFactory0.getEntryEncoding());
      assertEquals(0L, archiveOutputStream0.getBytesWritten());
      assertEquals(0, archiveOutputStream0.getCount());
      assertNotNull(archiveOutputStream0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("always");
      assertEquals("always", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
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
  public void test19()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      assertNull(archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      assertNotNull(bufferedInputStream0);
      
      try { 
        archiveStreamFactory0.createArchiveInputStream((InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not use reset and mark operations.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("$Zv=usa?");
      assertEquals("$Zv=usa?", archiveStreamFactory0.getEntryEncoding());
      assertNotNull(archiveStreamFactory0);
      
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      assertEquals(7, byteArrayInputStream0.available());
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(7, byteArray0.length);
      assertNotNull(byteArrayInputStream0);
      
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
}