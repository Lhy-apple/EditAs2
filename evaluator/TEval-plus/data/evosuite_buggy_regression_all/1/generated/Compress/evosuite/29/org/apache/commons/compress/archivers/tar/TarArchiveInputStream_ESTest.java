/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:26:07 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.util.Map;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveInputStream_ESTest extends TarArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, (-2198));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-35), 80, (String) null);
      tarArchiveInputStream0.close();
      assertEquals(80, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, (-2198));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-35), 80, (String) null);
      tarArchiveInputStream0.setAtEOF(false);
      assertEquals(80, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, (-2198));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-35), 80, (String) null);
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertEquals(80, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, (-2198));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-35), 80, (String) null);
      tarArchiveInputStream0.reset();
      assertEquals(80, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 420, (String) null);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 420, (String) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)48);
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
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-14), (-14));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, "SCHILY.devmajor");
      tarArchiveInputStream0.getNextEntry();
      int int0 = tarArchiveInputStream0.read(byteArray0, (-14), 32);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 1601);
      tarArchiveInputStream0.getCurrentEntry();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 17, (-487));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 99, (byte) (-35), "6C07");
      boolean boolean0 = tarArchiveInputStream0.markSupported();
      assertFalse(boolean0);
      assertEquals((-35), tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-1), (byte)67);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 2675, 0);
      tarArchiveInputStream0.mark(1894);
      assertEquals(0, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 420, (String) null);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(0, int0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, (-3557), "mtime");
      long long0 = tarArchiveInputStream0.skip((-3557));
      assertEquals(0L, long0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-23), (-23));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, "SCHILY.devmajor");
      tarArchiveInputStream0.getNextEntry();
      tarArchiveInputStream0.getNextEntry();
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 1601);
      TarArchiveInputStream tarArchiveInputStream1 = new TarArchiveInputStream(tarArchiveInputStream0, (-749));
      byte[] byteArray0 = tarArchiveInputStream1.getLongNameData();
      assertEquals(0, tarArchiveInputStream1.getCount());
      assertNull(byteArray0);
      assertEquals(512, tarArchiveInputStream1.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)60;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, (-2198));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-35), 80, (String) null);
      boolean boolean0 = tarArchiveInputStream0.isEOFRecord(byteArray0);
      assertFalse(boolean0);
      assertEquals(80, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-1), (byte)67);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 2675, 0);
      assertEquals(0, tarArchiveInputStream0.getRecordSize());
      
      byte[] byteArray1 = tarArchiveInputStream0.readRecord();
      assertNotNull(byteArray1);
      assertEquals(0, tarArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 1601);
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(tarArchiveInputStream0);
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (-1546), "x");
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 17, (-487));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 99, (byte) (-35), "6C07");
      tarArchiveInputStream0.skip((byte)23);
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
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-23), (-23));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, "SCHILY.devmajor");
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 420, (String) null);
      File file0 = MockFile.createTempFile("U<Xcb7Bl\"K", "zc*;<h86 (C");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0, "");
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches(byteArray0, 265);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 257
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      boolean boolean0 = TarArchiveInputStream.matches(byteArray0, 0);
      assertFalse(boolean0);
  }
}
