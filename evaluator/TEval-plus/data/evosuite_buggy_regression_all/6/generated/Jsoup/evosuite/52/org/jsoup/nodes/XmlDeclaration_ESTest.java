/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:35:36 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.XmlDeclaration;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlDeclaration_ESTest extends XmlDeclaration_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("ilp?)$#", "ilp?)$#", false);
      String string0 = xmlDeclaration0.name();
      assertEquals("ilp?)$#", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("!", "!", true);
      String string0 = xmlDeclaration0.toString();
      assertEquals("#declaration", xmlDeclaration0.nodeName());
      assertEquals("<!!>", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("xml", "xml", false);
      String string0 = xmlDeclaration0.getWholeDeclaration();
      assertEquals("xml", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("xml", "xml", false);
      xmlDeclaration0.attr("xml", "xml");
      xmlDeclaration0.attr("}!!qT~qE{WU$K", "}!!qT~qE{WU$K");
      String string0 = xmlDeclaration0.getWholeDeclaration();
      assertEquals("xml version=\"\" encoding=\"\"", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("!", "!", false);
      String string0 = xmlDeclaration0.toString();
      assertEquals("<?!>", string0);
      assertEquals("#declaration", xmlDeclaration0.nodeName());
  }
}
