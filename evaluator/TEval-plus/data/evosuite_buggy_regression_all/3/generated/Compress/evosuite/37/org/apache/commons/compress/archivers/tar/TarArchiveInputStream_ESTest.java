/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:29:48 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.time.ZoneId;
import java.util.Map;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveInputStream_ESTest extends TarArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      // Undeclared exception!
      try { 
        tarArchiveInputStream0.close();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 61, 0);
      byte[] byteArray1 = tarArchiveInputStream0.getLongNameData();
      assertNull(byteArray1);
      assertEquals(0, tarArchiveInputStream0.getRecordSize());
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0, 5060);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.reset();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, (-293), 0);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "<");
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, "GNU.sparse.realsize");
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.getCurrentEntry();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      TarArchiveInputStream tarArchiveInputStream1 = new TarArchiveInputStream(tarArchiveInputStream0, (-4170), "SCHILY.devmajor");
      assertEquals(512, tarArchiveInputStream1.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      boolean boolean0 = tarArchiveInputStream0.markSupported();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.mark(33188);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 643, (byte)0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 643, 643);
      tarArchiveInputStream0.skip(643);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("~hfS%N4GnBFgv}7");
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Truncated TAR archive
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      long long0 = tarArchiveInputStream0.skip(0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("~hfS%N4GnBFgv}7");
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("#aqQ7KN)1", true);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.setAtEOF(true);
      byte[] byteArray0 = tarArchiveInputStream0.getLongNameData();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertNotNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      TarArchiveInputStream tarArchiveInputStream1 = new TarArchiveInputStream(tarArchiveInputStream0);
      byte[] byteArray0 = tarArchiveInputStream1.getLongNameData();
      assertNull(byteArray0);
      assertEquals(512, tarArchiveInputStream1.getRecordSize());
      assertEquals(0L, tarArchiveInputStream1.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)18;
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, (byte)18, 653);
      boolean boolean0 = tarArchiveInputStream0.isEOFRecord(byteArray0);
      assertFalse(boolean0);
      assertEquals(653, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 0, 0);
      // Undeclared exception!
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // / by zero
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 643, (byte)0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 643, 643);
      tarArchiveInputStream0.skip(643);
      // Undeclared exception!
      try { 
        tarArchiveInputStream0.read();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No current tar entry
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("~hfS%N4GnBFgv}7");
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("~hfS%N4GnBFgv}7");
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      tarArchiveEntry0.fillStarSparseData(map0);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches(byteArray0, 26111);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 257
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = TarArchiveInputStream.matches(byteArray0, (byte)0);
      assertFalse(boolean0);
  }
}