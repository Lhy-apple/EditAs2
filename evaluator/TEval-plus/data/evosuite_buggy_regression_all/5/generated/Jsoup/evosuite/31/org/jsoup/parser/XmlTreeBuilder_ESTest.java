/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:13:46 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.Token;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlTreeBuilder_ESTest extends XmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.Comment token_Comment0 = new Token.Comment();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.process(token_Comment0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.Doctype token_Doctype0 = new Token.Doctype();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.process(token_Doctype0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      xmlTreeBuilder0.initialiseParse("#document", "#document", parseErrorList0);
      Token.StartTag token_StartTag0 = new Token.StartTag("Must be false");
      xmlTreeBuilder0.process(token_StartTag0);
      Token.EndTag token_EndTag0 = new Token.EndTag("#document");
      boolean boolean0 = xmlTreeBuilder0.process(token_EndTag0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("e-c}I", "e-c}I");
      Token.StartTag token_StartTag0 = new Token.StartTag("e-c}I");
      token_StartTag0.selfClosing = true;
      boolean boolean0 = xmlTreeBuilder0.process(token_StartTag0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("e-c}I", "e-c}I");
      Token.StartTag token_StartTag0 = new Token.StartTag("e-c}I");
      token_StartTag0.selfClosing = true;
      token_StartTag0.tagName = "address";
      boolean boolean0 = xmlTreeBuilder0.process(token_StartTag0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("e-c}I", "e-c}I");
      Token.EndTag token_EndTag0 = new Token.EndTag("e-c}I");
      boolean boolean0 = xmlTreeBuilder0.process(token_EndTag0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      xmlTreeBuilder0.initialiseParse("#document", "#document", parseErrorList0);
      Token.EndTag token_EndTag0 = new Token.EndTag("#document");
      boolean boolean0 = xmlTreeBuilder0.process(token_EndTag0);
      assertTrue(boolean0);
  }
}
