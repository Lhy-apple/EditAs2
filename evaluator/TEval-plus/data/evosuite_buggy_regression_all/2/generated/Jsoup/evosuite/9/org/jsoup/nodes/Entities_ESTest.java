/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:08:23 GMT 2023
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
      Document document0 = Document.createShell("?~K[&w.`#");
      Document.OutputSettings document_OutputSettings0 = document0.outputSettings();
      String string0 = Entities.escape("?~K[&w.`#", document_OutputSettings0);
      assertEquals("?~K[&amp;w.`#", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Entities entities0 = new Entities();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = Entities.unescape("?~K[&amp;w.`#");
      assertEquals("?~K[&w.`#", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      String string0 = Entities.unescape("tdot");
      assertEquals("tdot", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("g{N&#5");
      assertEquals("g{N\u0005", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("kP&zX=S");
      assertEquals("kP&zX=S", string0);
  }
}
