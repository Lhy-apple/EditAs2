/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:27:32 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.StringReader;
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
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.reset();
      assertEquals(0, tarArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.close();
      assertEquals(0, tarArchiveInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("W-r~K2=8%p5", (byte) (-42));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 31);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      assertEquals(1000, TarArchiveEntry.MILLIS_PER_SECOND);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)18, (byte)18);
      tarArchiveInputStream0.getNextEntry();
      TarArchiveEntry tarArchiveEntry0 = tarArchiveInputStream0.getNextTarEntry();
      assertNull(tarArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      TarArchiveEntry tarArchiveEntry0 = tarArchiveInputStream0.getCurrentEntry();
      assertNull(tarArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.setAtEOF(true);
      assertEquals(0, tarArchiveInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      long long0 = tarArchiveInputStream0.skip(0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      long long0 = tarArchiveInputStream0.skip(4596L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      long long0 = tarArchiveInputStream0.skip(40960L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)18;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.getNextTarEntry();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(0, tarArchiveInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.getNextTarEntry();
      assertEquals(0, byteArrayInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      StringReader stringReader0 = new StringReader("fZc");
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(stringReader0);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      StringReader stringReader0 = new StringReader("%ku.bdU >ts");
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(stringReader0);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)18, (byte)18);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("W-r~K2=8%p5", (byte) (-42));
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 31);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches((byte[]) null, 512);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      boolean boolean0 = TarArchiveInputStream.matches(byteArray0, 3);
      assertFalse(boolean0);
  }
}