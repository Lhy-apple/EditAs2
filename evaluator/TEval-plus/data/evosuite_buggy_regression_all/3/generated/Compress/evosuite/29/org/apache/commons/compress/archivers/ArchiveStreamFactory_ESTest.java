/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:27:58 GMT 2023
 */

package org.apache.commons.compress.archivers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileDescriptor;
import java.io.FilterOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.SequenceInputStream;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Enumeration;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.ar.ArArchiveOutputStream;
import org.apache.commons.compress.archivers.jar.JarArchiveOutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.archivers.zip.ZipArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
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
  public void test02()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      Enumeration<ObjectInputStream> enumeration0 = (Enumeration<ObjectInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) sequenceInputStream0);
      assertEquals(0L, archiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory((String) null);
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) mockFileInputStream0);
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((String) null, (InputStream) archiveInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Archivername must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream("u'nY73~FP$#31eYf,oK", (InputStream) null);
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
      PipedInputStream pipedInputStream0 = new PipedInputStream(694);
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("ar", (InputStream) pipedInputStream0);
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
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
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory(" [Gu<");
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
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ArchiveInputStream archiveInputStream0 = archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) pipedInputStream0);
      assertEquals(0, archiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("cpio");
      archiveStreamFactory0.createArchiveInputStream("zip", (InputStream) pipedInputStream0);
      assertEquals("cpio", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      Enumeration<ObjectInputStream> enumeration0 = (Enumeration<ObjectInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      archiveStreamFactory0.setEntryEncoding("vQ_iV");
      archiveStreamFactory0.createArchiveInputStream("jar", (InputStream) sequenceInputStream0);
      assertEquals("vQ_iV", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream(694);
      try { 
        archiveStreamFactory0.createArchiveInputStream("7z", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // The 7z doesn't support streaming.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("org.apache.commons.compress.archivers.zip.unsupportedzipfeatureexception");
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      archiveStreamFactory0.createArchiveInputStream("cpio", (InputStream) mockFileInputStream0);
      assertEquals("org.apache.commons.compress.archivers.zip.unsupportedzipfeatureexception", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        archiveStreamFactory0.createArchiveInputStream("dump", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Pipe not connected
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("cpio");
      try { 
        archiveStreamFactory0.createArchiveInputStream("dump", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Pipe not connected
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        archiveStreamFactory0.createArchiveInputStream("59[~KA@C2ppdpd~0", (InputStream) pipedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver: 59[~KA@C2ppdpd~0 not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ArArchiveOutputStream arArchiveOutputStream0 = (ArArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("ar", byteArrayOutputStream0);
      assertEquals(0, ArArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      MockPrintStream mockPrintStream0 = new MockPrintStream("ar");
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream((String) null, mockPrintStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Archivername must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("$?M]", (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // OutputStream must not be null.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = (TarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("tar", pipedOutputStream0);
      assertEquals(0, TarArchiveOutputStream.BIGNUMBER_ERROR);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      ZipArchiveOutputStream zipArchiveOutputStream0 = (ZipArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("zip", byteArrayOutputStream0);
      assertEquals("UTF8", zipArchiveOutputStream0.getEncoding());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("=");
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      FilterOutputStream filterOutputStream0 = new FilterOutputStream(pipedOutputStream0);
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveOutputStream("zip", filterOutputStream0);
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // =
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      JarArchiveOutputStream jarArchiveOutputStream0 = (JarArchiveOutputStream)archiveStreamFactory0.createArchiveOutputStream("jar", byteArrayOutputStream0);
      assertEquals((-1), ZipArchiveOutputStream.DEFAULT_COMPRESSION);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("zip");
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      archiveStreamFactory0.createArchiveOutputStream("tar", pipedOutputStream0);
      assertEquals("zip", archiveStreamFactory0.getEntryEncoding());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      try { 
        archiveStreamFactory0.createArchiveOutputStream("7z", byteArrayOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // The 7z doesn't support streaming.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory("=");
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      FilterOutputStream filterOutputStream0 = new FilterOutputStream(pipedOutputStream0);
      try { 
        archiveStreamFactory0.createArchiveOutputStream("", filterOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Archiver:  not found.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      Enumeration<ObjectInputStream> enumeration0 = (Enumeration<ObjectInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      // Undeclared exception!
      try { 
        archiveStreamFactory0.createArchiveInputStream((InputStream) sequenceInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Mark is not supported.
         //
         verifyException("org.apache.commons.compress.archivers.ArchiveStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
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
  public void test28()  throws Throwable  {
      ArchiveStreamFactory archiveStreamFactory0 = new ArchiveStreamFactory();
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
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