/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:15:04 GMT 2023
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
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration(" h", " h", false);
      String string0 = xmlDeclaration0.name();
      assertEquals(" h", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration(" h", " h", false);
      String string0 = xmlDeclaration0.toString();
      assertEquals("<? h>", string0);
      assertEquals("#declaration", xmlDeclaration0.nodeName());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("xml", "xml", true);
      xmlDeclaration0.attr("svg", "svg");
      XmlDeclaration xmlDeclaration1 = (XmlDeclaration)xmlDeclaration0.attr("xml", "xml");
      String string0 = xmlDeclaration1.getWholeDeclaration();
      assertEquals("xml version=\"\" encoding=\"\"", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("xml", "xml", true);
      String string0 = xmlDeclaration0.outerHtml();
      assertEquals("<!xml>", string0);
      assertEquals("#declaration", xmlDeclaration0.nodeName());
  }
}