/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:14:08 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.net.URI;
import java.time.ZoneId;
import java.util.Map;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.net.MockURI;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveInputStream_ESTest extends TarArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)124);
      tarArchiveInputStream0.close();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.setAtEOF(true);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 963, 963, (String) null);
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertFalse(boolean0);
      assertEquals(963, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(bufferedInputStream0, (-422), 512);
      tarArchiveInputStream0.reset();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 41);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      URI uRI0 = MockURI.aFileURI;
      MockFile mockFile0 = new MockFile(uRI0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
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
  public void test06()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, "GNU.sparse.realsize");
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      tarArchiveInputStream0.getCurrentEntry();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, (-2497), (String) null);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 2543);
      boolean boolean0 = tarArchiveInputStream0.markSupported();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 256, 256, "SCHILY.devminor");
      tarArchiveInputStream0.mark((byte)95);
      assertEquals(256, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(0, int0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)124);
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(0, int0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 2543);
      long long0 = tarArchiveInputStream0.skip((-1766L));
      assertEquals(0L, long0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      URI uRI0 = MockURI.aFileURI;
      URI uRI1 = MockURI.resolve(uRI0, "");
      MockFile mockFile0 = new MockFile(uRI1);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(10L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte) (-50);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-50), 2);
      // Undeclared exception!
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 99
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.getLongNameData();
      URI uRI0 = MockURI.aFileURI;
      MockFile mockFile0 = new MockFile(uRI0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(1L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)32;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 440);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, 0, 0);
      // Undeclared exception!
      try { 
        tarArchiveInputStream0.getNextTarEntry();
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
      byte[] byteArray0 = new byte[10];
      byteArray0[2] = (byte) (-77);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-77), 2);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(2L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      DataInputStream dataInputStream0 = new DataInputStream(pipedInputStream0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(dataInputStream0, (-58), (-2132));
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("");
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(zipArchiveEntry0);
      assertFalse(boolean0);
      assertEquals((-2132), tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)18);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)18, false);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertTrue(boolean0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)18);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)18, false);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      tarArchiveEntry0.fillStarSparseData(map0);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 2, 2);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(4L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches((byte[]) null, 2529);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      boolean boolean0 = TarArchiveInputStream.matches((byte[]) null, 255);
      assertFalse(boolean0);
  }
}
