/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:31:34 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.util.Map;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
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
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 561, 0, (String) null);
      assertEquals(0, tarArchiveInputStream0.getRecordSize());
      
      byte[] byteArray0 = tarArchiveInputStream0.getLongNameData();
      assertNull(byteArray0);
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.setAtEOF(true);
      byte[] byteArray0 = tarArchiveInputStream0.getLongNameData();
      assertNull(byteArray0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(257);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, (-1139));
      tarArchiveInputStream0.mark(32);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(183);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(pipedInputStream0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(bufferedInputStream0);
      tarArchiveInputStream0.reset();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-1), (byte)0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 6);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("F", false);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(6L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, "0");
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(183);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(pipedInputStream0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(bufferedInputStream0);
      TarArchiveEntry tarArchiveEntry0 = tarArchiveInputStream0.getCurrentEntry();
      assertNull(tarArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 33188, "xMY");
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 33188);
      boolean boolean0 = tarArchiveInputStream0.markSupported();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 0, 0, (String) null);
      long long0 = tarArchiveInputStream0.skip(0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte) (-117);
      boolean boolean0 = tarArchiveInputStream0.isEOFRecord(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(tarArchiveInputStream0);
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 1982);
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte)32;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, (byte)0);
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(257);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, (-1139));
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("T!C");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 33188);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches(byteArray0, 2608);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 257
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      boolean boolean0 = TarArchiveInputStream.matches((byte[]) null, (-73));
      assertFalse(boolean0);
  }
}