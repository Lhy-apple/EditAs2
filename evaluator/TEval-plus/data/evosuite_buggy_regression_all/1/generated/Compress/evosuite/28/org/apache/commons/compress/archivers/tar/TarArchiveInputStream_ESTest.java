/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:25:35 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.FileDescriptor;
import java.io.InputStream;
import java.io.PushbackInputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveInputStream_ESTest extends TarArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.close();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(", dateTimeAccessed=");
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(6, tarArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 98, 98);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.setAtEOF(true);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)0, (-832));
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertFalse(boolean0);
      assertEquals((-832), tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.reset();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (-1813), (byte) (-9));
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals((-9), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream((InputStream) null);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pushbackInputStream0, (String) null);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0, 56);
      tarArchiveInputStream0.getCurrentEntry();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 641, "OH");
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)0, (-832));
      int int0 = tarArchiveInputStream0.available();
      assertEquals(0, int0);
      assertEquals((-832), tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-11), (byte) (-11));
      TarArchiveInputStream tarArchiveInputStream1 = new TarArchiveInputStream(tarArchiveInputStream0, (byte) (-11), 0);
      tarArchiveInputStream1.getNextTarEntry();
      byte[] byteArray1 = tarArchiveInputStream1.getLongNameData();
      assertNull(byteArray1);
      assertEquals(0, tarArchiveInputStream1.getRecordSize());
      assertEquals(0L, tarArchiveInputStream1.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[0] = (byte)2;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)2, (byte)2);
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
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("+QX{3i*U( s,`\"IG");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.getNextEntry();
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)32;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)32, (byte)32);
      ByteArrayInputStream byteArrayInputStream1 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream1);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[2] = (byte)2;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)2, (byte)2);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(2, tarArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("+QX{3i*U( s,`\"IG");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, 16877);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)2, (byte)2);
      tarArchiveInputStream0.getNextTarEntry();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(3, tarArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches((byte[]) null, 3405);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      boolean boolean0 = TarArchiveInputStream.matches(byteArray0, (-3585));
      assertFalse(boolean0);
  }
}
