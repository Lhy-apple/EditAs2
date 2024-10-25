/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:47:57 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.SequenceInputStream;
import java.time.ZoneId;
import java.util.Enumeration;
import java.util.Map;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveInputStream_ESTest extends TarArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      tarArchiveInputStream0.close();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      DataInputStream dataInputStream0 = new DataInputStream(pipedInputStream0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(dataInputStream0);
      tarArchiveInputStream0.setAtEOF(false);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, (-1616), (-5));
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertEquals((-5), tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 998, "SCHILY.filetype");
      tarArchiveInputStream0.reset();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, Integer.MAX_VALUE);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("N~P");
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
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, "codeToEnum");
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, (-1616), (-23));
      tarArchiveInputStream0.getCurrentEntry();
      assertEquals((-23), tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.markSupported();
      assertFalse(boolean0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, Integer.MAX_VALUE);
      tarArchiveInputStream0.mark((byte) (-25));
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("TAPE");
      FileSystemHandling.createFolder(evoSuiteFile0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 998, "SCHILY.filetype");
      MockFile mockFile0 = new MockFile("TAPE");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "pp");
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      long long0 = tarArchiveInputStream0.skip(0L);
      assertEquals(0L, long0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("TAPE");
      FileSystemHandling.createFolder(evoSuiteFile0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 998, "SCHILY.filetype");
      MockFile mockFile0 = new MockFile("TAPE");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "pp");
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      long long0 = tarArchiveInputStream0.skip(33188);
      assertEquals(0L, long0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0, (-1616), 0);
      assertEquals(0, tarArchiveInputStream0.getRecordSize());
      
      tarArchiveInputStream0.getNextEntry();
      byte[] byteArray0 = tarArchiveInputStream0.getLongNameData();
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)16;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 1, 1);
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
      Enumeration<PipedInputStream> enumeration0 = (Enumeration<PipedInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("lC1iraO`,NixKknI", false);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(sequenceInputStream0, 1000);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      
      tarArchiveInputStream0.getNextTarEntry();
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      byte[] byteArray0 = tarArchiveInputStream0.getLongNameData();
      assertNotNull(byteArray0);
      assertEquals(0, tarArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)32);
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)32;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)32);
      Map<String, String> map0 = tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)16, 1);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(1L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("(i'>wnw7ty0})c=S");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("(i'>wnw7ty0})c=S");
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      tarArchiveEntry0.fillStarSparseData(map0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches((byte[]) null, 10240);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      boolean boolean0 = TarArchiveInputStream.matches(byteArray0, (-476));
      assertFalse(boolean0);
  }
}
