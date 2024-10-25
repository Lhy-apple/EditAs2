/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:25:47 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.util.HashMap;
import org.apache.commons.compress.archivers.arj.ArjArchiveEntry;
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
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      tarArchiveInputStream0.setAtEOF(true);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)31);
      MockFile mockFile0 = new MockFile("4gRf", "4gRf");
      mockFile0.mkdirs();
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(0, int0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.isAtEOF();
      assertFalse(boolean0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.reset();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      int int0 = tarArchiveInputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("3;nKZ,;R(]X~");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0, 16877, 16877);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream((InputStream) null, 814);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pushbackInputStream0, "H");
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
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null, 814, "H");
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      boolean boolean0 = tarArchiveInputStream0.markSupported();
      assertFalse(boolean0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.mark((byte) (-113));
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[13];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      int int0 = tarArchiveInputStream0.available();
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      long long0 = tarArchiveInputStream0.skip(0L);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      MockFile mockFile0 = new MockFile("");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      try { 
        tarArchiveInputStream0.getLongNameData();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[1] = (byte)8;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-62), 2);
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
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)31);
      MockFile mockFile0 = new MockFile("4gRf", "4gRf");
      tarArchiveInputStream0.getNextTarEntry();
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveInputStream0.setCurrentEntry(tarArchiveEntry0);
      tarArchiveInputStream0.getLongNameData();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[21];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte)0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[13];
      byteArray0[0] = (byte)32;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0);
      tarArchiveInputStream0.parsePaxHeaders(byteArrayInputStream0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[3] = (byte) (-4);
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(byteArrayInputStream0, (byte) (-4), 2);
      tarArchiveInputStream0.getNextTarEntry();
      assertEquals(2L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      ArjArchiveEntry arjArchiveEntry0 = new ArjArchiveEntry();
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(arjArchiveEntry0);
      assertFalse(boolean0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("Mlyp]uI%2|&y.6guI");
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("Mlyp]uI%2|&y.6guI");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      tarArchiveEntry0.fillStarSparseData(hashMap0);
      boolean boolean0 = tarArchiveInputStream0.canReadEntryData(tarArchiveEntry0);
      assertEquals(512, tarArchiveInputStream0.getRecordSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0, 53, 0);
      assertEquals(0, tarArchiveInputStream0.getRecordSize());
      
      byte[] byteArray0 = tarArchiveInputStream0.getLongNameData();
      assertNull(byteArray0);
      assertEquals(0L, tarArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarArchiveInputStream.matches((byte[]) null, 21589);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      boolean boolean0 = TarArchiveInputStream.matches((byte[]) null, (-1906));
      assertFalse(boolean0);
  }
}
