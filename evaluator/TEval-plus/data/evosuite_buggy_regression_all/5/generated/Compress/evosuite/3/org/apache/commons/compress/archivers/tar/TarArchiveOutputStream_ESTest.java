/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:48:43 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("O*<!)'.5d*T$&l!)", false);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0, 2166);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 100);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.flush();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 100);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 100);
      MockFile mockFile0 = new MockFile("  @t~<d6", "  @t~<d6");
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, "  @t~<d6");
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockFile mockFile0 = new MockFile("1qDelKeL<5*", "1qDelKeL<5*");
      File file0 = MockFile.createTempFile("1qDelKeL<5*", "org.a]ache.commons.compress.archiver6.cpio.CpioArchiveEntry", (File) mockFile0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 16877, 16877);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/lhy/TEval-plus/1qDelKeL<5*_/1qDelKeL<5*_/1qDelKeL<5*0org.a]ache.commons.compress.archiver6.cpio.CpioArchiveEntry' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFile mockFile0 = new MockFile("1qDelKeL<5*", "1qDelKeL<5*");
      File file0 = MockFile.createTempFile("1qDelKeL<5*", "org.a]ache.commons.compress.archiver6.cpio.CpioArchiveEntry", (File) mockFile0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 1000);
      tarArchiveOutputStream0.setLongFileMode(2);
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Output buffer is closed
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarBuffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockFile mockFile0 = new MockFile("1qDelKeL<5*", "1qDelKeL<5*");
      File file0 = MockFile.createTempFile("1qDelKeL<5*", "org.a]ache.commons.compress.archiver6.cpio.CpioArchiveEntry", (File) mockFile0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 16877, 16877);
      tarArchiveOutputStream0.setLongFileMode(1);
      // Undeclared exception!
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("O*<!)'.5d*T$&l!)");
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0, 33188);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(16877, tarArchiveEntry0.getMode());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.closeArchiveEntry();
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("O*<!)'.5d*T$&l!)", false);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("O*<!)'.5d*T$&l!)", (byte) (-34));
      tarArchiveEntry0.setSize(2166);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0, 2166);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      byte[] byteArray0 = new byte[1];
      tarArchiveOutputStream0.write(byteArray0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry 'O*<!)'.5d*T$&l!)' closed at '1' before the '2166' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 517, 517);
      TarArchiveOutputStream tarArchiveOutputStream1 = new TarArchiveOutputStream(tarArchiveOutputStream0, 271);
      try { 
        tarArchiveOutputStream1.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '271' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("O*<!)'.5d*T$&l!)", false);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("O*<!)'.5d*T$&l!)", (byte) (-34));
      tarArchiveEntry0.setSize(685L);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0, 2166);
      byte[] byteArray0 = new byte[9];
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      tarArchiveOutputStream0.write(byteArray0);
      tarArchiveOutputStream0.write(byteArray0);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("O*<!)'.5d*T$&l!)", false);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("O*<!)'.5d*T$&l!)", (byte) (-34));
      tarArchiveEntry0.setSize(685L);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0, 2166);
      byte[] byteArray0 = new byte[9];
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      tarArchiveOutputStream0.write(byteArray0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.write(byteArray0, 31, 512);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("O*<!)'.5d*T$&l!)", false);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("O*<!)'.5d*T$&l!)", (byte) (-34));
      tarArchiveEntry0.setSize(685L);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0, 571);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      byte[] byteArray0 = new byte[4];
      try { 
        tarArchiveOutputStream0.write(byteArray0, 0, 571);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // record has length '4' with offset '0' which is less than the record size of '512'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarBuffer", e);
      }
  }
}