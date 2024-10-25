/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:34:04 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.nio.charset.IllegalCharsetNameException;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.zip.ZipFile;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipFile_ESTest extends ZipFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      File file0 = MockFile.createTempFile("org.apache.commons.compress.archivers.zip.AsiExtraField", "", (File) null);
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(file0, "");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("VC[>N", "3'V_,cd");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // 3'V_,cd
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("@1t8#/@1t8#");
      FileSystemHandling.appendLineToFile(evoSuiteFile0, "archive's ZIP64 end of central directory locator is corrupt.");
      MockFile mockFile0 = new MockFile("@1t8#", "@1t8#");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(mockFile0);
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ZipFile.closeQuietly((ZipFile) null);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("@1t8#");
      byte[] byteArray0 = new byte[33];
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("@1t8#");
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }
}
