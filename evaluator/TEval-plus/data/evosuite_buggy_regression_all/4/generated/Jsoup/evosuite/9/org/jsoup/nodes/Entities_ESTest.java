/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:50:44 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Document document0 = Document.createShell("_<wX8j[e+");
      Document.OutputSettings document_OutputSettings0 = document0.outputSettings();
      String string0 = Entities.escape("_<wX8j[e+", document_OutputSettings0);
      assertEquals("_&lt;wX8j[e+", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Entities entities0 = new Entities();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = Entities.unescape("_&lt;wX8j[e+");
      assertEquals("_<wX8j[e+", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      String string0 = Entities.unescape("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("?&#4OEsu]UF");
      assertEquals("?\u0004OEsu]UF", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("?&#X4,Nu]UF");
      assertEquals("?\u0004,Nu]UF", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = Entities.unescape(";&K7Szd&jer");
      assertEquals(";&K7Szd&jer", string0);
  }
}