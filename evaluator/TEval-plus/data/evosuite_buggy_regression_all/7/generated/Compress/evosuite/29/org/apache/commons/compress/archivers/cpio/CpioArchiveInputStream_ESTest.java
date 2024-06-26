/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:36:06 GMT 2023
 */

package org.apache.commons.compress.archivers.cpio;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CpioArchiveInputStream_ESTest extends CpioArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0);
      try { 
        cpioArchiveInputStream0.getNextCPIOEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unknown magic [\u0000\u0000\u0000\u0000\u0000\u0000]. Occured at byte: 6
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, 56);
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, 56, 56);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      DataInputStream dataInputStream0 = new DataInputStream(mockFileInputStream0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(dataInputStream0);
      int int0 = cpioArchiveInputStream0.available();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0);
      cpioArchiveInputStream0.close();
      cpioArchiveInputStream0.close();
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0, (byte) (-37));
      cpioArchiveInputStream0.close();
      try { 
        cpioArchiveInputStream0.getNextCPIOEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0, (byte) (-37));
      byte[] byteArray0 = new byte[2];
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, (int) (byte) (-37), (int) (byte) (-37));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream((InputStream) null);
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read((byte[]) null, 1319, (-2081));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)46, 286);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (String) null);
      byte[] byteArray1 = new byte[0];
      int int0 = cpioArchiveInputStream0.read(byteArray1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream((InputStream) null);
      CpioArchiveInputStream cpioArchiveInputStream1 = new CpioArchiveInputStream(cpioArchiveInputStream0, 835);
      try { 
        cpioArchiveInputStream1.getNextEntry();
        fail("Expecting exception: EOFException");
      
      } catch(EOFException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)46, 286);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (String) null);
      long long0 = cpioArchiveInputStream0.skip((byte)46);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.skip((-1L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // negative skip length
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0);
      cpioArchiveInputStream0.skip(0L);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream((InputStream) null);
      cpioArchiveInputStream0.skip(2147483647L);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      CpioArchiveInputStream.matches(byteArray0, (byte)0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[0] = (byte)113;
      CpioArchiveInputStream.matches(byteArray0, 2863);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[1] = (byte)113;
      CpioArchiveInputStream.matches(byteArray0, 2863);
  }
}
