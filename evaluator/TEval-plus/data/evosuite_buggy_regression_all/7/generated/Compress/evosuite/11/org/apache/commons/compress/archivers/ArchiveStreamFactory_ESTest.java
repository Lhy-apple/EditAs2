/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:34:04 GMT 2023
 */

package org.apache.commons.compress.archivers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.ArchiveOutputStream;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.ar.ArArchiveOutputStream;
import org.apache.commons.compress.archivers.jar.JarArchiveOutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArchiveStreamFactory_ESTest extends ArchiveStreamFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("dump", (InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // InputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) byteArrayInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
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
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[20];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("ar", (InputStream) byteArrayInputStream0);
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) byteArrayInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) byteArrayInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) byteArrayInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
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
  public void test08()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        archiveStreamFactory0.createArchiveInputStream("Stream must not be null.", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: Stream must not be null. not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("q)_mtu", false);
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("cpio", mockFileOutputStream0);
      assertEquals(0L, archiveOutputStream0.getBytesWritten());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream((String) null, pipedOutputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Archivername must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("ar", (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // OutputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ArArchiveOutputStream arArchiveOutputStream0 = (ArArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("ar", byteArrayOutputStream0);
      assertEquals(1, ArArchiveOutputStream.LONGFILE_BSD);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("zip", byteArrayOutputStream0);
      assertEquals(0, archiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("cpio");
      TarArchiveOutputStream tarArchiveOutputStream0 = (TarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("tar", mockFileOutputStream0);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("cpio");
      JarArchiveOutputStream jarArchiveOutputStream0 = (JarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("jar", mockFileOutputStream0);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockPrintStream mockPrintStream0 = new MockPrintStream("jar");
      try { 
        archiveStreamFactory0.createArchiveOutputStream("#G* F>4=CIhyS-|", mockPrintStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: #G* F>4=CIhyS-| not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream((InputStream) byteArrayInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
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
      PipedInputStream pipedInputStream0 = new PipedInputStream();
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
