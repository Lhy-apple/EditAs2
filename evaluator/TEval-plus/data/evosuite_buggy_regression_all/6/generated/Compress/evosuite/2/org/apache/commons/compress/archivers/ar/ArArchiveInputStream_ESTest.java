/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:11:02 GMT 2023
 */

package org.apache.commons.compress.archivers.ar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.SequenceInputStream;
import java.util.Enumeration;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ar.ArArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArArchiveInputStream_ESTest extends ArArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(byteArrayInputStream0);
      arArchiveInputStream0.skip(13L);
      try { 
        arArchiveInputStream0.getNextEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // failed to read entry header
         //
         verifyException("org.apache.commons.compress.archivers.ar.ArArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(byteArrayInputStream0);
      try { 
        arArchiveInputStream0.getNextEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // failed to read header
         //
         verifyException("org.apache.commons.compress.archivers.ar.ArArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[19];
      byteArray0[0] = (byte)33;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(byteArrayInputStream0);
      try { 
        arArchiveInputStream0.getNextArEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // invalid header !\u0000\u0000\u0000\u0000\u0000\u0000\u0000
         //
         verifyException("org.apache.commons.compress.archivers.ar.ArArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(byteArrayInputStream0);
      arArchiveInputStream0.read(byteArray0);
      ArchiveEntry archiveEntry0 = arArchiveInputStream0.getNextEntry();
      assertNull(archiveEntry0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(byteArrayInputStream0);
      arArchiveInputStream0.skip(8L);
      try { 
        arArchiveInputStream0.getNextEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // failed to read entry header
         //
         verifyException("org.apache.commons.compress.archivers.ar.ArArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Enumeration<ObjectInputStream> enumeration0 = (Enumeration<ObjectInputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false, false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(sequenceInputStream0);
      arArchiveInputStream0.close();
      arArchiveInputStream0.close();
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      byte[] byteArray0 = new byte[19];
      byteArray0[0] = (byte)33;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ArArchiveInputStream arArchiveInputStream0 = new ArArchiveInputStream(byteArrayInputStream0);
      int int0 = arArchiveInputStream0.read();
      assertEquals(33, int0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      byte[] byteArray0 = new byte[19];
      boolean boolean0 = ArArchiveInputStream.matches(byteArray0, (byte)33);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      boolean boolean0 = ArArchiveInputStream.matches(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      byte[] byteArray0 = new byte[19];
      byteArray0[0] = (byte)33;
      boolean boolean0 = ArArchiveInputStream.matches(byteArray0, (byte)33);
      assertFalse(boolean0);
  }
}