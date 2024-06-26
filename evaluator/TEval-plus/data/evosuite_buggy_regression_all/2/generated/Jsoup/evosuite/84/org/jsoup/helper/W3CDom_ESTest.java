/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:14:44 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.helper.W3CDom;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;
import org.w3c.dom.DOMException;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class W3CDom_ESTest extends W3CDom_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("Tag cannot be sef clwring; no a void tag");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      DataNode dataNode0 = new DataNode("Tag cannot be sef clwring; no a void tag");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(dataNode0, 91);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.W3CDom$W3CBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      String string0 = w3CDom0.asString((org.w3c.dom.Document) null);
      assertEquals("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Parser.parse("S8!+:+Gu<!", "");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder((org.w3c.dom.Document) null);
      w3CDom_W3CBuilder0.head((Node) null, (-1658));
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Parser.parse("Tag cannot be self closing; not a void tag", "Tag cannot be self closing; not a void tag");
      Charset charset0 = Charset.defaultCharset();
      document0.charset(charset0);
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Parser.parseBodyFragment("Tag cannot Ne self closing; not a void tag", "Tag cannot Ne self closing; not a void tag");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      document0.attr("xmlns", "Tag cannot Ne self closing; not a void tag");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(document0, (-2266));
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Document document0 = Document.createShell(", ");
      Element element0 = document0.attr("xmlns:value", "xmlns:value");
      W3CDom w3CDom0 = new W3CDom();
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(element0, (-1721724269));
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Parser.parseBodyFragment("JUlY!+c:", "JUlY!+c:");
      document0.prependElement("JUlY!+c:");
      // Undeclared exception!
      try { 
        w3CDom0.fromJsoup(document0);
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }
}
