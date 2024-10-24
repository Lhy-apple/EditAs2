/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:47:28 GMT 2023
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
      Document document0 = new Document("rmoue'\",he");
      Document.OutputSettings document_OutputSettings0 = document0.new OutputSettings();
      String string0 = Entities.escape("rmoue'\",he", document_OutputSettings0);
      assertEquals("rmoue'&quot;,he", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Entities entities0 = new Entities();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = Entities.unescape("g7bx9#c^|M&#2:qpZ");
      assertEquals("g7bx9#c^|M\u0002:qpZ", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      String string0 = Entities.unescape("rmoustache");
      assertEquals("rmoustache", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("dibJ&pWd8");
      assertEquals("dibJ&pWd8", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("-9bJ&pr\"d*p^");
      assertEquals("-9bJ\u227A\"d*p^", string0);
  }
}
