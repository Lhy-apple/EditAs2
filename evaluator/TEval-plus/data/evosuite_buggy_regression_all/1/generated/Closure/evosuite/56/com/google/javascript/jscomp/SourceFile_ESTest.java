/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:08:28 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.SourceFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.StringReader;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SourceFile_ESTest extends SourceFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromGenerator("\n*ulo", (SourceFile.Generator) null);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      SourceFile.Generated sourceFile_Generated0 = new SourceFile.Generated("`ZaoXPPN4[r?{$+", sourceFile_Generator0);
      sourceFile_Generated0.clearCachedSource();
      assertFalse(sourceFile_Generated0.isExtern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      SourceFile sourceFile0 = SourceFile.fromFile(" <hr", charset0);
      // Undeclared exception!
      try { 
        sourceFile0.getLineOffset(0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected line number between 1 and 1
         // Actual: 0
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile("%s (%s) must be less than size (%s)", "*9HL|Qq$dzHj");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      sourceFile_OnDisk0.clearCachedSource();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StringReader stringReader0 = new StringReader("%0x=Yr");
      SourceFile sourceFile0 = SourceFile.fromReader("%0x=Yr", stringReader0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("_8T#", "_8T#", "_8T#");
      sourceFile_Preloaded0.clearCachedSource();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      SourceFile.Generated sourceFile_Generated0 = new SourceFile.Generated("J83|=", sourceFile_Generator0);
      sourceFile_Generated0.getCodeNoCache();
      assertFalse(sourceFile_Generated0.isExtern());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("lnIPIs@k", "\nActual: ");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(mockFile0);
      sourceFile_OnDisk0.getName();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("com.google.javascript.jscomp.SourceFile$Generated");
      String string0 = sourceFile0.getOriginalPath();
      assertFalse(sourceFile0.isExtern());
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = (SourceFile.Preloaded)SourceFile.fromCode("lnIPIs@k", "lnIPIs@k", "lnIPIs@k");
      sourceFile_Preloaded0.setIsExtern(false);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      File file0 = MockFile.createTempFile("%s (%s) must be less than size (%s)", "*9HL|Qq$dzHj");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      sourceFile_OnDisk0.toString();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(269);
      try { 
        SourceFile.fromInputStream("", "", (InputStream) pipedInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        SourceFile.fromInputStream("com.google.javascript.jscomp.SourceFile$OnDisk", (InputStream) pipedInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockFile mockFile0 = new MockFile("h+J~mA3k}e.8e6Y#");
      SourceFile sourceFile0 = SourceFile.fromFile((File) mockFile0);
      boolean boolean0 = sourceFile0.isExtern();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        SourceFile.fromCode((String) null, (String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      // Undeclared exception!
      try { 
        SourceFile.fromFile("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = (SourceFile.Preloaded)SourceFile.fromCode("UTF-16BE", "J83|=", "J83|=");
      sourceFile_Preloaded0.getNumLines();
      // Undeclared exception!
      try { 
        sourceFile_Preloaded0.getLineOffset(1530);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected line number between 1 and 1
         // Actual: 1530
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\n*ulo", "\n*ulo", "\n*ulo");
      int int0 = sourceFile_Preloaded0.getLineOffset(1);
      assertFalse(sourceFile_Preloaded0.isExtern());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("o", "zxB=JZ_I", "zxB=JZ_I");
      sourceFile_Preloaded0.getNumLines();
      int int0 = sourceFile_Preloaded0.getNumLines();
      assertFalse(sourceFile_Preloaded0.isExtern());
      assertEquals(1, int0);
      assertEquals("zxB=JZ_I", sourceFile_Preloaded0.getOriginalPath());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromCode("lnIPIs@k", "lnIPIs@k", "lnIPIs@k");
      sourceFile0.getOriginalPath();
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("RS0%T");
      try { 
        sourceFile0.getCodeReader();
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      File file0 = MockFile.createTempFile("%0x=zYr", "%s (%s) must not be greater than size (%s)");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      Region region0 = sourceFile_OnDisk0.getRegion(3900);
      assertNull(region0);
      
      sourceFile_OnDisk0.getCodeReader();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\ncul ", "\ncul ", "\ncul ");
      String string0 = sourceFile_Preloaded0.getLine((-6));
      assertNotNull(string0);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\ncul ", "\ncul ", "\ncul ");
      String string0 = sourceFile_Preloaded0.getLine(22);
      assertFalse(sourceFile_Preloaded0.isExtern());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      File file0 = MockFile.createTempFile("%0x=Yr", "%0x=Yr");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      sourceFile_OnDisk0.getLine((-2812));
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nAcual ", "\nAcual ");
      Region region0 = sourceFile_Preloaded0.getRegion(101);
      assertNull(region0);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\n", "\n");
      Region region0 = sourceFile_Preloaded0.getRegion((-3532));
      assertNotNull(region0);
      assertEquals(1, region0.getBeginningLineNumber());
      assertEquals(2, region0.getEndingLineNumber());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn("J83|=").when(sourceFile_Generator0).getCode();
      SourceFile.Generated sourceFile_Generated0 = new SourceFile.Generated("J83|=", sourceFile_Generator0);
      sourceFile_Generated0.getRegion((-1528));
      Region region0 = sourceFile_Generated0.getRegion((-1528));
      assertNotNull(region0);
      assertFalse(sourceFile_Generated0.isExtern());
      assertEquals("J83|=", region0.getSourceExcerpt());
      assertEquals(1, region0.getEndingLineNumber());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MockFile mockFile0 = new MockFile("Expected line number between 1 and ");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(mockFile0, (Charset) null);
      assertFalse(sourceFile_OnDisk0.isExtern());
  }
}
