/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:13:43 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.helper.W3CDom;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Node;
import org.junit.runner.RunWith;
import org.w3c.dom.DOMException;
import org.w3c.dom.Document;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class W3CDom_ESTest extends W3CDom_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      String string0 = w3CDom0.asString((Document) null);
      assertEquals("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("a");
      document0.title("a");
      w3CDom0.fromJsoup(document0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = new org.jsoup.nodes.Document("</");
      document0.normalise();
      DataNode dataNode0 = new DataNode("</");
      Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(dataNode0, (-1));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.W3CDom$W3CBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = new org.jsoup.nodes.Document("</");
      document0.normalise();
      Comment comment0 = new Comment("</", "</");
      Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(comment0, (-158656234));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.W3CDom$W3CBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("org.jsoup.nodes.Node");
      Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      w3CDom_W3CBuilder0.head((Node) null, (-1));
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("");
      Charset charset0 = Charset.defaultCharset();
      document0.charset(charset0);
      Document document1 = w3CDom0.fromJsoup(document0);
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("pk>U>:=,Cd");
      document0.prependElement("pk>U>:=,Cd");
      // Undeclared exception!
      try { 
        w3CDom0.fromJsoup(document0);
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }
}