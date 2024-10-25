/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:09:23 GMT 2023
 */

package org.apache.commons.compress.archivers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Enumeration;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.ArchiveOutputStream;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.ar.ArArchiveOutputStream;
import org.apache.commons.compress.archivers.jar.JarArchiveOutputStream;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
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
      archiveStreamFactory0.setEntryEncoding("dump");
      assertEquals("dump", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("JYm@*V+hr2?Q");
      // Undeclared exception!
      try { 
        archiveStreamFactory0.setEntryEncoding("d(=2wKJ");
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
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) byteArrayInputStream0);
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
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory(" \u0000");
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((String) null, (InputStream) pipedInputStream0);
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
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("");
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      archiveStreamFactory0.createArchiveInputStream("ar", (InputStream) pipedInputStream0);
      assertEquals("", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        archiveStreamFactory0.createArchiveInputStream("arj", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Pipe not connected
         //
         verifyException("org.apache.commons.compress.archivers.arj.ArjArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("tar");
      try { 
        archiveStreamFactory0.createArchiveInputStream("arj", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Pipe not connected
         //
         verifyException("org.apache.commons.compress.archivers.arj.ArjArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) pipedInputStream0);
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory(" \u0000");
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) pipedInputStream0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         //  \u0000
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        archiveStreamFactory0.createArchiveInputStream("urfgrAN", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: urfgrAN not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory(" \u0000");
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("tar", (InputStream) pipedInputStream0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         //  \u0000
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("");
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) pipedInputStream0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      File file0 = MockFile.createTempFile("dump", "cpio");
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) mockFileInputStream0);
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream((InputStream) null);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) pushbackInputStream0);
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      try { 
        archiveStreamFactory0.createArchiveInputStream("dump", (InputStream) sequenceInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // unexpected EOF
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("arj");
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
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
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
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
  public void test19()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory(" \u0000");
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("zip", pipedOutputStream0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         //  \u0000
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("cpio", true);
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("tar", mockFileOutputStream0);
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream((String) null, archiveOutputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Archivername must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("", (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // OutputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      ArArchiveOutputStream arArchiveOutputStream0 = (ArArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("ar", pipedOutputStream0);
      assertEquals(1, ArArchiveOutputStream.LONGFILE_BSD);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = (ZipArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("zip", pipedOutputStream0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("tar", true);
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("cpio", mockFileOutputStream0);
      assertEquals(0L, archiveOutputStream0.getBytesWritten());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("ar", true);
      JarArchiveOutputStream jarArchiveOutputStream0 = (JarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("jar", mockFileOutputStream0);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("8\"^b");
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      try { 
        archiveStreamFactory0.createArchiveOutputStream("8\"^b", mockFileOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: 8\"^b not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("cpio", true);
      ArchiveOutputStream archiveOutputStream0 = archiveStreamFactory0.createArchiveOutputStream("tar", mockFileOutputStream0);
      ArchiveStreamFactory archiveStreamFactory1 = new ArchiveStreamFactory("tar");
      archiveStreamFactory1.createArchiveOutputStream("cpio", archiveOutputStream0);
      assertEquals("tar", archiveStreamFactory1.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("Failed to read entry: ", false);
      try { 
        archiveStreamFactory0.createArchiveOutputStream("7z", mockFileOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // The 7z doesn't support streaming.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
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
  public void test30()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
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
