/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:14:56 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Node;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Parser;
import org.jsoup.parser.TreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      Parser parser1 = parser0.setTreeBuilder((TreeBuilder) null);
      assertFalse(parser1.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String string0 = Parser.unescapeEntities("", false);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      ParseSettings parseSettings0 = new ParseSettings(false, false);
      Parser parser1 = parser0.settings(parseSettings0);
      assertFalse(parser1.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      List<Node> list0 = Parser.parseXmlFragment("", "");
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      parser0.getTreeBuilder();
      assertFalse(parser0.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Parser.parseBodyFragmentRelaxed("JFP]#77QE@`q;^!6%.J", "");
      assertEquals("", document0.location());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      assertFalse(parser0.isTrackErrors());
      
      Parser parser1 = parser0.setTrackErrors(1406);
      parser1.parseInput("eM]N.", "*v/!{V2.Ca_.Y8");
      assertTrue(parser0.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      parser0.getErrors();
      assertFalse(parser0.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      parser0.settings();
      assertFalse(parser0.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      parser0.parseInput("];`", "];`");
      assertFalse(parser0.isTrackErrors());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<html>\n <head></head>\n <body>\n  JFP]#77QE@`q;^!6%.J\n </body>\n</html>", "JFP]#77QE@`q;^!6%.J");
      assertEquals("JFP]#77QE@`q;^!6%.J", document0.baseUri());
  }
}
