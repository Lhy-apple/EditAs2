/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:28:28 GMT 2023
 */

package org.apache.commons.compress.archivers.cpio;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.FileDescriptor;
import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PushbackInputStream;
import org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CpioArchiveInputStream_ESTest extends CpioArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, (byte)103);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0, (String) null);
      assertEquals(0L, cpioArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.skip((-1673L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // negative skip length
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 1684, (byte)26);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte)26);
      try { 
        cpioArchiveInputStream0.getNextEntry();
        fail("Expecting exception: EOFException");
      
      } catch(EOFException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 1684, (byte)26);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte)26);
      int int0 = cpioArchiveInputStream0.available();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 4, 1331);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, 55);
      long long0 = cpioArchiveInputStream0.skip((byte)4);
      assertEquals(0L, long0);
      
      int int0 = cpioArchiveInputStream0.available();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(pipedInputStream0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pushbackInputStream0);
      cpioArchiveInputStream0.close();
      try { 
        cpioArchiveInputStream0.skip((byte)100);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte)0, "Px53-t");
      try { 
        cpioArchiveInputStream0.getNextEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unknown magic [\u0000\u0000\u0000\u0000\u0000\u0000]. Occured at byte: 6
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(pipedInputStream0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pushbackInputStream0);
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, (-1), (int) (byte)100);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(mockFileInputStream0, (byte) (-1));
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, (int) (byte)16, (int) (byte) (-1));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      byte[] byteArray0 = new byte[3];
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(pipedInputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveInputStream0.read(byteArray0, (int) (byte)28, 1380);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte)114, (String) null);
      cpioArchiveInputStream0.read(byteArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-128), (byte)0);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, (byte) (-47));
      cpioArchiveInputStream0.skip((byte)0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)103, (byte)103);
      CpioArchiveInputStream cpioArchiveInputStream0 = new CpioArchiveInputStream(byteArrayInputStream0, 1352);
      cpioArchiveInputStream0.skip(65286L);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      CpioArchiveInputStream.matches(byteArray0, (byte)26);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      CpioArchiveInputStream.matches(byteArray0, (byte)0);
  }
}